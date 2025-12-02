import csv
import itertools
import torch
from matplotlib import pyplot as plt
from scipy.io import savemat

from network import Network
from utils import evaluate, new_P, target_distribution, cross_view_consistency_loss, orthogonal_loss, \
    gaussian_kernel_matrix, compute_repel_loss, orthogonality_loss
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from Loda_data import load_data
import os
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

Dataname = 'HW'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--pre_epochs", default=500)
parser.add_argument("--tune_epochs", default=10000)  # 50
parser.add_argument("--feature_dim", default=10)
parser.add_argument("--UpdateCoo", default=1000)
parser.add_argument("--Pre_train", default=True)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 10  # 10 114514 3407
if args.dataset == 'ORL':
    args.batch_size = 256
PATH = './results/'
path = PATH + Dataname
args.save_dir = path
print("begin")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def pretrain():
    criterion = torch.nn.MSELoss()
    epoch = 0
    while epoch <= args.pre_epochs:
        model.train()
        tot_loss = 0.
        for batch_idx, (xs, _, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            label = torch.full((xs[0].size(0), args.n_clusters), 1.0 / args.n_clusters).to(device)
            optimizer.zero_grad()
            _, xrs, _, _, _ = model(xs, label)
            loss_list = []
            # print(f"[Debug] xrs shape: {xrs[0].shape}")
            for v in range(view):
                loss_list.append(criterion(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

        epoch += 1

    torch.save(model.state_dict(), args.save_dir + '/pretrain_weights.pth')


def fine_tune(global_sl, kl_w, ep_w):
    x = dataset.x
    y = dataset.labels
    view = args.view

    args.ARtime = 2

    x_gpu = [x_v.to(device) for x_v in x]
    global_sl_gpu = global_sl.to(device)

    y_pred_v = []
    with torch.no_grad():
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
        features = [[] for _ in range(view)]
        model.eval()

        for i in range(0, x[0].shape[0], args.batch_size):
            x_batch = [x_gpu[v][i:i + args.batch_size] for v in range(view)]
            p_batch = global_sl_gpu[i:i + args.batch_size]
            _, _, emb, _, _ = model(x_batch, p_batch)
            for v in range(view):
                features[v].append(emb[v].cpu())  

        features = [torch.cat(fea_list, dim=0) for fea_list in features]
        for v in range(view):
            y_pred_v.append(kmeans.fit_predict(features[v].numpy()))  

    y_pred_sp = []
    for v in range(view):
        acc, nmi, ari, pur, f = evaluate(y, y_pred_v[v])
        print('Start-' + str(v + 1) + ': acc=%.5f, nmi=%.5f, f=%.5f' % (acc, nmi, f))
        y_pred_sp.append(y_pred_v[v])

    print("deep cluster stage")

    ite = 0
    aligment_large = 0
    center_init = 0
    KL_function = torch.nn.KLDivLoss(reduction='batchmean')
    MSE_function = torch.nn.MSELoss()
    Losses = []

    ACC_res = []
    NMI_res = []
    ARI_res = []
    best_model_state = None
    best_acc = -np.inf
    best_nmi = -np.inf
    best_ari = -np.inf

    while True:
        # update target distribution
        if ite % args.UpdateCoo == 0:
            with torch.no_grad():
                model.eval()
                features = [[] for _ in range(view)]
                q_all = [[] for _ in range(view)]

                for i in range(0, x[0].shape[0], args.batch_size):
                    x_batch = [x_gpu[v][i:i + args.batch_size] for v in range(view)]
                    p_batch = global_sl_gpu[i:i + args.batch_size]
                    q, _, emb, _, _ = model(x_batch, p_batch)
                    for v in range(view):
                        q_all[v].append(q[v])
                        features[v].append(emb[v])

                q_all = [torch.cat(qv, dim=0).cpu() for qv in q_all]
                features = [torch.cat(fea_list, dim=0).cpu() for fea_list in features]

                for v in range(view):
                    y_pred_sp[v] = q_all[v].argmax(1)

                z = np.hstack([features[v].numpy() for v in range(view)])  # Ö±½Ó×ªnumpy

                kmean = KMeans(n_clusters=args.n_clusters, n_init=100)
                y_pred = kmean.fit_predict(z)

            scale = len(y)
            for i in range(len(y)):
                predict = y_pred_sp[0][i]
                for v in range(view - 1):
                    if predict == y_pred_sp[v + 1][i]:
                        continue
                    else:
                        scale -= 1
                        break
            alignment = (scale / len(y))
            print('Aligned Ratio: %.2f%%. %d' % (alignment * 100, len(y)))
            if alignment > 0.90:
                aligment_large += 1
            else:
                aligment_large = 0
            if aligment_large < args.ARtime:
                Center_init = kmean.cluster_centers_
                new_p = new_P(z, Center_init)
                p = target_distribution(new_p)
                p = p.to(device)
                global_sl_gpu = p.detach().clone().to(device)
                center_init += 1
            else:
                break

        # train
        tot_loss = 0.
        recon_l = 0.
        kl_l = 0.
        batch_indices = list(range(0, len(x[0]), args.batch_size))

        for batch_idx, start_idx in enumerate(batch_indices):
            end_idx = min(start_idx + args.batch_size, len(x[0]))
            xs_batch = [x_gpu[v][start_idx:end_idx] for v in range(len(x))]
            p_batch = global_sl_gpu[start_idx:end_idx]

            optimizer.zero_grad()
            q, x_bar, encoder, mu_stack, gate_probs = model(xs_batch, p_batch)

            kl_loss = 0.
            recon_loss = 0.
            entropy_loss = 0.
            orth_loss = 0.
            for v in range(len(xs_batch)):
                recon_loss += MSE_function(xs_batch[v], x_bar[v])
                kl_loss += KL_function(torch.log(q[v]), p_batch)
                gate_v = gate_probs[v]
                entropy_loss += - (gate_v * (gate_v + 1e-8).log()).sum(dim=1).mean()

            orth_loss = orthogonal_loss(mu_stack)

            loss = kl_w * kl_loss + 1 * recon_loss + ep_w * (entropy_loss + orth_loss)
            tot_loss += loss.item()
            loss.backward()
            optimizer.step()

        if ite % 100 == 0:
            print('Epoch {}'.format(ite), 'Loss:{:.6f}'.format(tot_loss / len(batch_indices)))
            model.eval()
            q_all = [[] for _ in range(view)]

            with torch.no_grad():
                for i in range(0, x[0].shape[0], args.batch_size):
                    x_batch = [x_gpu[v][i:i + args.batch_size] for v in range(view)]
                    p_label = global_sl_gpu[i:i + args.batch_size]
                    q_batch, _, _, _, _ = model(x_batch, p_label)
                    for v in range(view):
                        q_all[v].append(q_batch[v])

            
            q_all_views = [torch.cat(q_list, dim=0).cpu() for q_list in q_all]
            y_pred = [q.argmax(dim=1).numpy() for q in q_all_views]
            y_q = sum(q_all_views)
            y_mean_pred = y_q.argmax(dim=1).numpy()

            for v in range(view):
                acc, nmi, ari, pur, fscore = evaluate(y, y_pred[v])

            acc_all, nmi_all, ari_all, pur_all, fscore_all = evaluate(y, y_mean_pred)

            if best_acc < acc_all:
                best_acc = acc_all
                best_nmi = nmi_all
                best_ari = ari_all
                # best_model_state = model.state_dict().copy()

        Losses.append(tot_loss / len(batch_indices))
        ite += 1
        if ite >= int(args.tune_epochs):
            break

    # if best_model_state is not None:
    #     model_dir = f'./results/{args.dataset}/model_path/'
    #     os.makedirs(model_dir, exist_ok=True)
    #
    #     model_path = os.path.join(model_dir, f'best_model_{kl_w}_{ep_w}.pth')
    #     torch.save(best_model_state, model_path)

    return best_acc, best_nmi, best_ari


if not os.path.exists('./results'):
    os.makedirs('./results')

T = 1
for i in range(T):
    setup_seed(seed)
    dataset = load_data(args.dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    view = dataset.views
    view_shapes = []
    args.view = view
    for v in range(view):
        view_shapes.append(dataset.x[v].shape[1])
    n_clusters = len(np.unique(dataset.labels))
    args.n_clusters = n_clusters
    x = dataset.x
    y = dataset.labels
    sp = [0.001, 0.01, 0.1, 1, 10, 100]
    results = []
    iter = 1

    for kl in range(len(sp)):
	kl_w = sp[kl]
	
	for ep in range(len(sp)):
	    ep_w = sp[ep]
	  
	    print('----------')
	    print('kl_w:', kl_w)
	    print('ep_w:', ep_w)
	    print('----------')

	    ACCList = np.zeros((iter, 1))
	    NMIList = np.zeros((iter, 1))
	    ARIList = np.zeros((iter, 1))
	    ACC_MEAN = np.zeros((1, 2))
	    NMI_MEAN = np.zeros((1, 2))
	    ARI_MEAN = np.zeros((1, 2))

	    for it in range(iter):
		model = Network(input_dim=view_shapes, feature_dim=args.feature_dim, view_shape=view,
		                alpha=args.alpha,
		                clusters=n_clusters, device=device)
		model = model.to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
		                             weight_decay=args.weight_decay)
		criterion = Loss(args.batch_size, n_clusters, args.temperature_l, device).to(device)

		args.Pre_train = False
		if args.Pre_train:
		    pretrain()
		else:
		    model.load_state_dict(torch.load(args.save_dir + '/pretrain_weights.pth'))

		global_sl = torch.full((x[0].size(0), args.n_clusters), 1.0 / args.n_clusters).to(device)

		best_acc, best_nmi, best_ari = fine_tune(global_sl, kl_w, ep_w)

		ACCList[it, :] = best_acc
		NMIList[it, :] = best_nmi
		ARIList[it, :] = best_ari

	    ACC_MEAN[0, :] = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
	    NMI_MEAN[0, :] = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
	    ARI_MEAN[0, :] = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)

	    with open('./results/' + args.dataset + '/' + 'result_without.txt', 'a') as f:
		f.write(args.dataset + '_woall_Result:' + '\n')
		f.write('kl Loss:' + str(kl_w) + '\n')
		f.write('expert Loss:' + str(ep_w) + '\n')
		f.write('ACC_MEAN:' + str(ACC_MEAN[0][0] * 100) + '(' + str(ACC_MEAN[0][1] * 100) + ')\n')
		f.write('NMI_MEAN:' + str(NMI_MEAN[0][0] * 100) + '(' + str(NMI_MEAN[0][1] * 100) + ')\n')
		f.write('ARI_MEAN:' + str(ARI_MEAN[0][0] * 100) + '(' + str(ARI_MEAN[0][1] * 100) + ')\n')
		f.write('\n')



