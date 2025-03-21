import numpy as np
import pytransform3d.rotations as rotations
import h5py
import cv2
import os
import re

class Adaptor:

    def pose6D2quat(self, pose:np.ndarray):
        
        column_1 = pose[:3]
        column_2 = pose[3:]

        R = np.column_stack((column_1, column_2, np.cross(column_1, column_2)))

        # quat = rotations.quaternion_from_matrix(R)
        euler = rotations.euler_from_matrix(R, 0,1,2, extrinsic=True)
        return euler

    def read_data_in_batches(self, hdf5_file, batch_size=100):
        """
        逐批读取数据的生成器。按需读取数据块而非一次性加载。
        """
        total_data = len(hdf5_file['qpos'])
        for i in range(0, total_data, batch_size):
            batch_qpos = hdf5_file['qpos'][i:i + batch_size]
            batch_images = hdf5_file['images']['cam_high'][i:i + batch_size]
            yield batch_qpos, batch_images
            
    def qpos_2_ee_pose(self, qpos:np.ndarray):

        # r_joint_pos = qpos[0:10]
        # l_joint_pos = qpos[50:60]

        # l_gripper_joint_pos = qpos[60:65]
        # r_gripper_joint_pos = qpos[25:30]

        l_pose6d = qpos[83:89]
        r_pose6d = qpos[33:39]
        l_quat = self.pose6D2quat(l_pose6d)
        r_quat = self.pose6D2quat(r_pose6d)
        l_ee_trans = qpos[80:83]
        r_ee_trans = qpos[30:33]
        # import pdb
        # pdb.set_trace()
        return np.concatenate((l_ee_trans, l_quat, r_ee_trans, r_quat))

    def package_ee_pose_action(self, data_num:int, ee_pose:np.ndarray, action_chunk:int):
        '''
        ee_pose: <number , 12>
        '''
        action_xyz = np.zeros((data_num,action_chunk,6))
        # print(f"ee_pose size:{ee_pose.size}")
        for i in range(data_num):
            if(i < data_num-action_chunk):
                action_xyz[i][0:action_chunk] = ee_pose[i+1:i+1+action_chunk]
            else:
                action_xyz[i][0:data_num-i-1] = ee_pose[i+1:data_num]
                for j in range(data_num-i-1, action_chunk):
                    action_xyz[i][j] = ee_pose[-1]
        return action_xyz
    
    def package_joint_pos_action(self,data_num:int,joint_pos:np.ndarray,action_chunk:int):
        '''
        joint_positions: <number, 7>
        '''
        action_joint = np.zeros((data_num,action_chunk,14))
        for i in range(data_num):
            if(i < data_num-action_chunk):
                action_joint[i][0:action_chunk] = joint_pos[i+1:i+1+action_chunk]
            else:
                action_joint[i][0:data_num-i-1] = joint_pos[i+1:data_num]
                for j in range(data_num-i-1, action_chunk):
                    action_joint[i][j] = joint_pos[-1]                
        return action_joint

    def qpos_2_joint_positions(self, qpos:np.ndarray):

        l_joint_pos = qpos[50:56]
        r_joint_pos = qpos[0:6]
        l_gripper_pos = np.array([qpos[60]])
        r_gripper_pos = np.array([qpos[10]])

        l_pos = np.concatenate((l_joint_pos,l_gripper_pos))
        r_pos = np.concatenate((r_joint_pos,r_gripper_pos))

        return np.concatenate((l_pos,r_pos))


    def cam_high_2_front_img(self, cam:np.ndarray):
        '''
        input: (59606,)
        output: (480,640,3)
        '''
        frame = cv2.imdecode(cam, cv2.IMREAD_COLOR)
        
        return frame
    
    def resample_to_100_frames(self, data, target_length=100):
        current_length = data
        
        if current_length < target_length:
            # 如果当前长度小于目标长度，重复最后一帧
            len = target_length - current_length
            padded_data = np.tile([current_length-1], len)
            # padded_data = np.tile(last_frame, (target_length - current_length, 1) + (1,) * (data.ndim - 1))
            return np.concatenate([np.linspace(0, current_length - 1, current_length, dtype=int), padded_data], axis=0)
        
        # 计算均匀抽帧的间隔
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        # resampled_data = data[indices]
        
        return indices


    def rdt2ego_add_demo(self, rdt_data:h5py.Group, ego_data:h5py.Group, inst):
        '''
        egoMinic demo Structure:
        demo_i
            actions_joints_act
            actions_xyz_act
            obs
                ee_pose
                front_img_1
                front_img_1_line
                joint_positions
                right_wrist_img

        rdt data Structure:
        observations
            images
                cam_high
                cam_left_wrist
                cam_right_wrist
            qpos
            qvel
        '''

        total_frame = len(rdt_data['qpos'])
        data_num = 400
        print(f"current file data numbers:{total_frame}")

        # create a new demo group
        obs_group = ego_data.create_group('observations')

        # create 'ee_pose' dataset with compression and chunking
        ee_pose = np.zeros((data_num, 12))

        # create 'front_img_1' dataset with compression and chunking
        front_img = np.zeros((data_num, 480, 640, 3))
        right_wrist_img = np.zeros((data_num, 480, 640, 3))
        left_wrist_img = np.zeros((data_num, 480, 640, 3))
        
        # create joint position dataset
        joint_pos = np.zeros((data_num, 14))

        indices = self.resample_to_100_frames(total_frame, 400)
        for j in range (len(indices)):
            i = indices[j]
            ee_pose[j]  = self.qpos_2_ee_pose(rdt_data['qpos'][i])
            joint_pos[j] = self.qpos_2_joint_positions(rdt_data['qpos'][i])
            front_img[j] = self.cam_high_2_front_img(rdt_data['images']['cam_high'][i])
            left_wrist_img[j] = self.cam_high_2_front_img(rdt_data['images']['cam_left_wrist'][i])
            right_wrist_img[j] = self.cam_high_2_front_img(rdt_data['images']['cam_right_wrist'][i])

        # create actions
        # action_xyz = self.package_ee_pose_action(data_num, ee_pose, action_chunk)
        # action_joints_pos = self.package_joint_pos_action(data_num, joint_pos, action_chunk)
        
        # create obs
        images = obs_group.create_group("images")
        images.create_dataset(
            "cam_high", data=front_img)
        obs_group.create_dataset(
            "joint_positions", data=joint_pos)
        images.create_dataset(
            "cam_right_wrist", data=right_wrist_img)
        images.create_dataset(
            "cam_left_wrist", data=left_wrist_img)
        obs_group.create_dataset(
            "qpos", data=ee_pose)
        print("obs_finish")
        
        # create actions datasets with compression
        ego_data.create_dataset("actions_xyz_act", data=ee_pose)
        ego_data.create_dataset("actions_joints_act", data=joint_pos)
        ego_data.create_dataset("action", data=joint_pos)
        ego_data.create_dataset("language_raw", data=inst)
        print("actions_finish")
        

    # 递归搜索指定路径下的所有符合条件的文件
    def find_episode_files(self,root_dir):
        # 用于存储匹配文件的路径
        episode_files = []

        # 遍历根目录下的所有文件和文件夹
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # 遍历当前文件夹中的每个文件
            for filename in filenames:
                # 检查文件名是否匹配 'episode_xx.hdf5' 模式
                if filename.startswith('episode_') and filename.endswith('.hdf5'):
                    # 获取文件的绝对路径
                    file_path = os.path.join(dirpath, filename)
                    episode_files.append(file_path)
        return episode_files
    
    def rdt2ego(self, ego_data_path:str, rdt_data_path:str="/mnt/hpfs/baaiei/lvhuaihai/agilex_data/cobot/2025.03.18.17.32"):
        # get filename of all episode
        files_list = self.find_episode_files(rdt_data_path)

        # create a new dataset to store the new data
        os.makedirs(ego_data_path, exist_ok=True)
        
        with open(os.path.join(rdt_data_path, "inst.txt"), "r") as file:
            lines = file.readlines()
            cleaned_lines = [re.sub(r'^\d+\.\s*', '', line.rstrip('\n')) for line in lines]
            
            for i in range(len(files_list)):
                print(f"start save in episode_{i}.hdf5")
                print("start to handle {}_th file: ".format(i) + files_list[i])
                ego_group:h5py.File = h5py.File(os.path.join(ego_data_path, f"episode_{i}.hdf5"), 'w')

                rdt_data:h5py.File = h5py.File(files_list[i], 'r')
                rdt_group = rdt_data['observations']

                self.rdt2ego_add_demo(rdt_group, ego_group, cleaned_lines)
                ego_group.attrs["num_samples"] = 400
                ego_group.close()
                rdt_data.close()
        


        # mask_group = ego_data.create_group("mask")
        
        # # 创建 train 和 valid 数据集
        # train_data = []
        # valid_data = []

        # all_indices = list(range(len(ego_data['data'])))

        # import random
        # train_indices = random.sample(all_indices, 10)
        # valid_indices = list(set(all_indices) - set(train_indices))  # 剩下的分配到 valid

        # # 按照选定的索引填充 train 和 valid 数据
        # for i in train_indices:
        #     demo_str = f"demo_{i}"
        #     train_data.append(demo_str)

        # for i in valid_indices:
        #     demo_str = f"demo_{i}"
        #     valid_data.append(demo_str)

        # # 创建数据集并写入字符串
        # mask_group.create_dataset("train", data=[s.encode('utf-8') for s in train_data])
        # mask_group.create_dataset("valid", data=[s.encode('utf-8') for s in valid_data])

        # ego_data.close()
        

# 1. num_samples is corresponding to the fixed length of each demo
# 2. action chunk setting
# 3. rename the file according to the pairs
# 4. indices division

if __name__ == "__main__":
    adaptor = Adaptor()
    action_chunk = 50
    adaptor.rdt2ego(f"/mnt/hpfs/baaiei/lvhuaihai/agilex_data/vla_cobot/groceries_agliex_white_longbread_cobot_VLA")
    # 路径名字：demo name; pairs number; action chunk; fixed length of each demo;
    # num_samples = fixed length of each demo
    # self.resample_to_100_frames修改参数为fixed length of each demo
    