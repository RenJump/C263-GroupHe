import torch

import argparse
import warnings
import time

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample

warnings.filterwarnings('ignore')


def main(args):
    dataname = args.dataname
    device = args.device
    steps = args.steps
    save_path = args.save_path
    save_path = save_path[:-4]
    sample_mode = args.sample_mode

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)

    if sample_mode == 0:
        in_dim = train_z.shape[1]

        mean = train_z.mean(0)

        denoise_fn = MLPDiffusion(in_dim, 1024).to(device)

        model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)

        model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))

        '''
            Generating samples    
        '''
        start_time = time.time()

        '''num_samples = train_z.shape[0]
        print(num_samples)
        sample_dim = in_dim
    
        x_next = sample(model.denoise_fn_D, num_samples, sample_dim)
        x_next = x_next * 2 + mean.to(device)'''

        num_samples = train_z.shape[0]
        sample_dim = train_z.shape[1]  # Assuming in_dim corresponds to the shape of train_z

        # 计算每个批次的大小
        batch_size = num_samples // 10

        for i in range(10):
            print(i)
            start_idx = i * batch_size
            if i == 9:
                end_idx = num_samples
            else:
                end_idx = start_idx + batch_size

            batch_z = train_z[start_idx:end_idx]

            x_next_batch = sample(model.denoise_fn_D, batch_z.shape[0], sample_dim)

            x_next_batch = x_next_batch * 2 + mean.to(device)

            x_next = x_next_batch

            syn_data = x_next.float().cpu().numpy()
            syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device)
            print(syn_num.shape)
            syn_df = recover_data(syn_num, syn_cat, syn_target, info)
            print(syn_df)
            idx_name_mapping = info['idx_name_mapping']
            idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

            syn_df.rename(columns=idx_name_mapping, inplace=True)

            syn_df.to_csv(save_path + '_' + str(i) + '.csv', index=False)

            end_time = time.time()
            print('Time:', end_time - start_time)

            print('Saving sampled data to {}'.format(save_path))

    elif sample_mode == 1:
        num_samples = train_z.shape[0]
        sample_dim = train_z.shape[1]  # Assuming in_dim corresponds to the shape of train_z

        batch_size = num_samples // 10

        total_df = pd.DataFrame()
        for i in range(10):
            print(i)
            start_idx = i * batch_size
            if i == 9:
                end_idx = num_samples
            else:
                end_idx = start_idx + batch_size

            x_next = train_z[start_idx:end_idx]

            syn_data = x_next.float().cpu().numpy()
            syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device)
            print(syn_num.shape)
            syn_df = recover_data(syn_num, syn_cat, syn_target, info)
            print(syn_df)
            idx_name_mapping = info['idx_name_mapping']
            idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

            syn_df.rename(columns=idx_name_mapping, inplace=True)

            total_df = pd.concat([total_df, syn_df], ignore_index=True)

        total_df.to_csv(save_path + '_step' + '.csv', index=False)
        print('Saving sampled data to {}'.format(save_path))
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'