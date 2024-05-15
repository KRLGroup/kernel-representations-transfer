import torch

import utils
from model import ACModel

class Agent:

    def __init__(self, env, obs_space, action_space, model_dir, progression_mode,
                device=None, argmax=False, num_envs=1, legacy_kernel_encoder=False,
                load_best_status=False):
        try:
            # print(model_dir)
            if not load_best_status:
                status = utils.get_status(model_dir)
                check = utils.storage.file_md5(utils.get_status_path(model_dir), mode='rb')
                print(f"[Agent] Loading trained model ({status['num_frames']} frames, checksum: {check})")
            else:
                status = utils.get_best_status(model_dir)
                check = utils.storage.file_md5(utils.get_best_status_path(model_dir), mode='rb')
                print(f"[Agent] Loading best trained model ({status['num_frames']} frames, checksum: {check})")
        except OSError:
            if load_best_status:
                print(f'[Agent] WARNING: Could NOT load best status from {model_dir}!')
            status = {"num_frames": 0, "update": 0}
        self.loaded_status = status

        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(env, progression_mode)

        self.acmodel = ACModel(env, obs_space, action_space, legacy_kernel_encoder=legacy_kernel_encoder)

        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        if not load_best_status:
            self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        else:
            self.acmodel.load_state_dict(status["model_state"])
        self.acmodel.to(self.device)
        self.acmodel.eval()


    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():
            dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]
