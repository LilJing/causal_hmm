from __future__ import print_function
import torch.utils.data
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def X_kl_loss_function(recon_x, x, mu, logvar, mu_prior, logvar_prior, args):

    BCE = torch.mean((recon_x - x)**2)
    KLD = 0.5 * torch.mean(torch.mean(logvar.exp()/logvar_prior.exp() +
                                     (mu - mu_prior).pow(2)/logvar_prior.exp() + logvar_prior - logvar - 1, 1))
    return BCE, KLD * args.kl_weight

def A_kl_loss_function(recon_A, A, mu, logvar, mu_prior, logvar_prior, args):

    BCE = torch.mean((recon_A - A)**2)
    KLD = 0.5 * torch.mean(torch.mean(logvar.exp()/logvar_prior.exp() +
                                     (mu - mu_prior).pow(2)/logvar_prior.exp() + logvar_prior - logvar - 1, 1))
    return BCE, KLD * args.kl_weight


def val(args, causal_hmm_model, classifier, val_dataset, batch_size, writer, epoch, n_iter, val_log):

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, drop_last=False, shuffle=False)

    rec_loss_x, kl_loss_x, rec_loss_A, kl_loss_A, pre_loss = 0, 0, 0, 0, 0

    causal_hmm_model.eval(), classifier.eval()
    with torch.no_grad():
        all_gt_y, all_pre_prob, all_pre_id = [], [], []
        for i_batch, (sample_batched) in enumerate(val_loader):

            image_data = torch.tensor(sample_batched[0])
            gt_y = torch.tensor(sample_batched[1])
            A_data = torch.tensor(sample_batched[2])
            B_data = torch.tensor(sample_batched[3])

            image_data = image_data.cuda()
            A_data = A_data.cuda()
            B_data = B_data.cuda()
            gt_y = gt_y.cuda()

            batch_size = gt_y.shape[0]
            z0 = torch.zeros(batch_size, args.z_size).cuda()
            s0 = torch.zeros(batch_size, args.s_size).cuda()
            v0 = torch.zeros(batch_size, args.v_size).cuda()
            B_size = 16
            B0 = torch.zeros(batch_size, B_size).cuda()

            z_t_last, s_t_last, v_t_last = z0, s0, v0
            all_rec_loss, all_kl_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda()
            all_rec_loss_A, all_kl_loss_A = torch.zeros(1).cuda(), torch.zeros(1).cuda()

            grade_num = args.to_grade - args.from_grade
            for grade in range(args.to_grade - args.from_grade):
                if grade == 0:
                    X_t = image_data[:, grade, :, :, :]
                    A_t = A_data[:, grade, :]
                    B_t_last = B0
                    z_t_last, s_t_last, v_t_last = z0, s0, v0
                else:
                    X_t = image_data[:, grade, :, :, :]
                    A_t = A_data[:, grade, :]
                    B_t_last = B_data[:, grade, :]

                vae_rec_x_t, vae_rec_A, mu_h, logvar_h, \
                mu_h_prior, logvar_h_prior, mu_z_t, mu_s_t, mu_v_t, logvar_v, mu_v_prior, logvar_v_prior = \
                    causal_hmm_model(X_t, A_t, B_t_last, z_t_last, s_t_last, v_t_last, test=True)

                loss_re_x_t, loss_kl_x_t = X_kl_loss_function(vae_rec_x_t, X_t, mu_h, logvar_h, mu_h_prior, logvar_h_prior,
                                                              args)
                loss_re_A_t, loss_kl_A_t = A_kl_loss_function(vae_rec_A, A_t, mu_v_t, logvar_v, mu_v_prior, logvar_v_prior,
                                                              args)

                all_rec_loss += loss_re_x_t
                all_kl_loss += loss_kl_x_t
                all_rec_loss_A += loss_re_A_t
                all_kl_loss_A += loss_kl_A_t

                z_t_last, s_t_last, v_t_last = mu_z_t, mu_s_t, mu_v_t

            classifier_input = torch.cat((s_t_last, v_t_last), 1)

            mean_rec_loss = all_rec_loss / grade_num
            mean_kl_loss = all_kl_loss / grade_num

            mean_rec_loss_A = all_rec_loss_A / grade_num
            mean_kl_loss_A = all_kl_loss_A / grade_num

            rec_loss_x += mean_rec_loss
            kl_loss_x += mean_kl_loss

            rec_loss_A += mean_rec_loss_A
            kl_loss_A += mean_kl_loss_A

            writer.add_scalar('val/mean rec loss of image', mean_rec_loss, n_iter)
            writer.add_scalar('val/mean kl loss of image', mean_kl_loss, n_iter)
            writer.add_scalar('val/mean rec loss of A', mean_rec_loss_A, n_iter)
            writer.add_scalar('val/mean kl loss of A', mean_kl_loss_A, n_iter)

            logit = classifier(classifier_input)
            func = nn.Softmax(dim=1)
            prob = func(logit)
            prob1 = prob[:, 1].cpu().detach().numpy()
            all_gt_y.extend(gt_y.cpu().numpy())
            all_pre_prob.extend(prob1)
            all_pre_id.extend(torch.max(logit, 1)[1].data.cpu().numpy())

            cn_loss = nn.CrossEntropyLoss()
            loss_pre = cn_loss(logit, gt_y.long())

            writer.add_scalar('val/classification loss', loss_pre, n_iter)
            pre_loss += loss_pre.item()

            n_iter += 1

            torch.cuda.empty_cache()

        all_gt_y , all_pre_prob, all_pre_id = \
            np.array(all_gt_y).flatten(), np.array(all_pre_prob).flatten(), np.array(all_pre_id).flatten()
        fpr, tpr, thresholds = roc_curve(all_gt_y, all_pre_prob, pos_label=1)
        all_val_auc = auc(fpr, tpr)
        all_val_acc = (all_pre_id == all_gt_y).sum() / len(all_gt_y)
        writer.add_scalar('val/AUC', all_val_auc, epoch)
        writer.add_scalar('val/ACC', all_val_acc, epoch)
        val_log['val_log'].info("val acc:{0}, val auc:{1} at epoch {2}".format(all_val_acc, all_val_auc, epoch))

        return all_val_auc
