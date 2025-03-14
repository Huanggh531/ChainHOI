import os
import os.path as osp
from os.path import join as pjoin
import argparse
import random
from tqdm import tqdm
import numpy as np

import codecs as cs


from multiprocessing import Pool, Queue, Manager

def calculate_distance2(data, i, j):
    n = j - i + 1
    ret = 0
    for idx in range(1, n):
        pa = data[i] + (idx / n) * (data[j] - data[i])
        # print(pa -data[idx+i])
        # ret = max(ret, np.abs(pa - data[idx+i]).max())
        ret = ret + np.abs(pa - data[idx + i]).max()
        # ret = ret + np.abs(pa - data[idx+i]).sum()
        # ret += np.linalg.norm(pa - data[idx+i], ord="inf")
    return ret

def init_table2(data, dis):
    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            dis[0, i, j] = calculate_distance2(data, i, j)
    return dis

def KF_selection2(data, KF_num=3):
    dis = np.zeros([KF_num+1, data.shape[0], data.shape[0]]) + 1e9
    dis = init_table2(data, dis)
    kfs = {}

    for n in range(1, KF_num+1):
        for i in range(0, data.shape[0]):
            for j in range(i + 1, data.shape[0]):
                # dis[n, i, j] = dis[n-1, i, j]
                # kfs["%d,%d,%d" % (n, i, j)] = [i, j]
                for k in range(i + 1, j):
                    if dis[n, i, j] > dis[n - 1, i, k] + dis[0, k, j]:
                        dis[n, i, j] = dis[n - 1, i, k] + dis[0, k, j]
                        if n == 1:
                            kfs["%d,%d,%d" % (n, i, j)] = [k, ]
                        else:
                            kfs["%d,%d,%d" % (n, i, j)] = kfs["%d,%d,%d" % (n - 1, i, k)] + [k, ]

    return dis, kfs

def worker(name, root_dir, min_motion_length, data_queue, num_keyframes):
    data_dict = {}
    try:
        motion = np.load(pjoin(root_dir, "new_joint_vecs", name + ".npy"), allow_pickle=True)
        if (len(motion)) < min_motion_length or (len(motion) >= 200):
            print(f"skip {name}, len {len(motion)}")
            return

        raw_motion = np.load(pjoin(root_dir, "new_joints", name + ".npy"), allow_pickle=True) # keyframe selection
        if len(raw_motion) != len(motion):
            print(name, len(motion), len(raw_motion))
            return

        text_data = []
        flag = False
        with (cs.open(pjoin(root_dir, "texts", name + ".txt")) as f):
            for j, line in enumerate(f.readlines()):
                text_dict = {}
                line_split = line.strip().split("#")
                caption = line_split[0]
                tokens = line_split[1].split(" ")
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                text_dict["caption"] = caption
                text_dict["tokens"] = tokens
                if f_tag == 0.0 and to_tag == 0.0:
                    flag = True
                    text_data.append(text_dict)
                else:
                    try:
                        n_motion = motion[int(f_tag * 20):int(to_tag * 20)]
                        n_raw_motion = raw_motion[int(f_tag * 20):int(to_tag * 20)]
                        if (len(n_motion)) < min_motion_length or ((len(n_motion) >= 200)):
                            continue
                        new_name = (random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name)
                        while new_name in data_dict:
                            new_name = (random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name)

                        _, kfs = KF_selection2(n_raw_motion, KF_num=num_keyframes - 1)
                        keyframes = kfs[f"{num_keyframes-1},0,{len(n_motion) - 1}"]
                        keyframes.append(len(n_motion) - 1)

                        data_dict[new_name] = {
                            "motion": n_motion,
                            "length": len(n_motion),
                            "text": [text_dict],
                            "keyframe_id": f"{name}_{j}", # for test
                            "selected_keyframe": keyframes
                        }
                    except:
                        print(f"error1! {name}")
                        pass

        if flag:
            _, kfs = KF_selection2(raw_motion, KF_num=num_keyframes - 1)
            keyframes = kfs[f"{num_keyframes-1},0,{len(raw_motion) - 1}"]
            keyframes.append(len(raw_motion) - 1)
            data_dict[name] = {
                "motion": motion,
                "length": len(motion),
                "text": text_data,
                "keyframe_id": f"{name}",
                "selected_keyframe": keyframes
            }
    except:
        print(f"error2! {name}")
        pass

    data_queue.put(data_dict)

def process(data_dir, split, min_motion_length, args):
    print(f"processing {split}...")
    with cs.open(osp.join(data_dir, split+".txt"), "r") as f:
        id_list = [line.strip() for line in f.readlines()]
    results = {}

    with Manager() as manager:
        output_queue = manager.Queue()
        pool = Pool(processes=args.num_workers)

        with tqdm(total=len(id_list)) as pbar:
            for x in id_list:
                pool.apply_async(worker,
                                 (x, data_dir, min_motion_length, output_queue, args.num_keyframes),
                                 callback=lambda _: pbar.update(1))

            pool.close()
            pool.join()

        while not output_queue.empty():
            ret = output_queue.get()
            results.update(ret)
    np.save(osp.join(data_dir, "data_"+split+".npy"), results)

def main(args):
    if args.dataset == "hml3d":
        data_dir = "./data/HumanML3D_guo/"
        min_motion_length = 40
    elif args.dataset == "kit":
        data_dir = "./data/KIT/"
        min_motion_length = 24
    else:
        raise ValueError(f"{args.dataset} not supported!")

    process(data_dir, "train", min_motion_length, args)
    process(data_dir, "val", min_motion_length, args)
    process(data_dir, "test", min_motion_length, args)

    print(f"dataset {args.dataset} preparation done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hml3d", required=True)
    parser.add_argument("--num_keyframes", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=64)
    args = parser.parse_args()
    main(args)