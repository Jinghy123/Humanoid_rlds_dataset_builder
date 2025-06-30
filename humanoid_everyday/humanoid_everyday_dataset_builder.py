from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class HumanoidDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for humanoid dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    
     def _load_depth_lzma(self, depth_lzma_path):
        with open(depth_lzma_path, "rb") as f:
            compressed_data = f.read()
            decompressed = lzma.decompress(compressed_data)
            depth_array = np.frombuffer(decompressed, dtype=np.uint16).reshape(
                (480, 640)
            )
        return depth_array
    
    def _load_lidar_points(self, lidar_path):
        def pad_to_six(m):
            whole, dec = m.group("whole"), m.group("dec")
            return f"{whole}.{dec.ljust(6, '0')}"
        
        def clean_and_sample_lidar(pcd_points, target_n=1500):
            np.random.seed(42)
            valid_mask = ~np.all(pcd_points == 0, axis=1)
            valid_points = pcd_points[valid_mask]
            num_valid = valid_points.shape[0]

            if num_valid >= target_n:
                indices = np.random.choice(num_valid, target_n, replace=False)
                fixed_points = valid_points[indices]
            else:
                pad_size = target_n - num_valid
                pad = np.zeros((pad_size, 3), dtype=np.float32)
                fixed_points = np.concatenate([valid_points, pad], axis=0)
            return fixed_points.astype(np.float32)

        lidar_path = re.sub(
            r"(?P<whole>\d+)\.(?P<dec>\d{1,6})(?=\.pcd$)", pad_to_six, lidar_path
        )
        pcd = o3d.io.read_point_cloud(lidar_path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        cleaned_pts = clean_and_sample_lidar(pts, target_n=1500)
        return cleaned_pts

    def _load_tactile(self, sensor_list):
        readings = []
        for sensor in sensor_list:
            readings.extend(sensor['usable_readings'])
        return np.array(readings, dtype=np.float32)


    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpg',
                            doc='Main camera RGB observation.',
                        ),
                        'depth': tfds.features.Tensor(
                            shape=(480, 640),
                            dtype=np.uint16,
                            doc='Realsense camera Depth observation.'
                        ),
                        'lidar': tfds.features.Tensor(
                            shape=(1500, 3),
                            dtype=np.float32,
                            doc='point cloud read from Livox Lidar sensor.'
                        ),
                        'state': tfds.features.Tensor(
                            shape=(28,),
                            dtype=np.float32,
                            doc='Robot state, consists of [14x robot joint angles, '
                                '14x hand position].',
                        ),
                        'tactile': tfds.features.Tensor(
                            shape=(66,),
                            dtype=np.float32,
                            doc='Tactile sensor readings, consists of [12x 4-dim A sensors,'
                                '6x 3-dim B sensors].'
                        ),
                    }),
                    'action': tfds.features.FeaturesDict({
                        'joint_pos': tfds.features.Tensor(
                            shape=(28,),
                            dtype=np.float32,
                            doc='Robot state, consists of [14x robot joint angles, '
                            '14x hand position].'
                        ),
                        'torque': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float32,
                            doc='Applied joint torques for 14 robot joints.'
                        ),
                    }),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/episode_*'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            base_dir = os.path.dirname(episode_path)
            data_path = os.path.join(episode_path, 'data.json')
            metadata_path = os.path.join(base_dir, 'metadata/metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            description_text = metadata['description']

            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                img_path = os.path.join(episode_path, step['image'])
                depth_path = os.path.join(episode_path, step['depth'])
                lidar_path = os.path.join(episode_path, step['lidar'])

                combined_states = np.array(step['states']['arm_state'] + step['states']['hand_state'], dtype=np.float32)
                depth_array = self._load_depth_lzma(depth_path)
                lidar_array = self._load_lidar_points(lidar_path)

                tactile_sensor_list = step['states']['hand_pressure_state']
                tactile_array = self._load_tactile(tactile_sensor_list)

                combined_joint_pos = np.array(step['actions']['sol_q'] + step['actions']['left_angles'] + step['actions']['right_angles'], dtype=np.float32)
                arm_torque = np.array(step['actions']['tau_ff'], dtype=np.float32)


                # compute Kona language embedding
                language_embedding = self._embed([description_text])[0].numpy()

                episode.append({
                    'observation': {
                        'image': img_path,
                        'depth': depth_array,
                        'lidar': lidar_array,
                        'state': combined_states,
                        'tactile': tactile_array,
                    },
                    'action': {
                        'joint_pos': combined_joint_pos,
                        'torque': arm_torque,
                    },
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': description_text,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

