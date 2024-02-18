import os.path as osp
from evaluator import Eval_thread
from dataloader import EvalDataset


def evaluate(save_test_path_root, save_dir, data_root, test_paths):

    pred_dir = save_test_path_root
    output_dir = save_dir
    gt_dir = data_root
    method_names = ["RGB_VST"]

    threads = []
    for dataset_setname in test_paths:
        

        dataset_name = dataset_setname.split('/')[0]

        for method in method_names:

            pred_dir_all = osp.join(pred_dir, dataset_name)
            print(pred_dir_all)

            if dataset_name == 'DUTS':
                gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname)) + '/DUTS-TE-Mask'
            else:
                gt_dir_all = '../VST/RGB_VST/Data/frames/GT'

            loader = EvalDataset(pred_dir_all, gt_dir_all)
            thread = Eval_thread(loader, method, dataset_setname, output_dir, cuda=True)
            threads.append(thread)
    for thread in threads:
        print(thread.run())

def main():
    
    evaluate(save_test_path_root="preds", save_dir="results", data_root="Data", test_paths=["frames"])

if __name__ == "__main__":
    main()