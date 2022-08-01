import os
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from utils.summary_tools import write_summary
from option_agent.hierarchical_policy import get_final_state_value, get_return_array, get_advantage


class Option_Agent(object):
    def __init__(self, args):
        self.args = args

        self.pi_func = None
        self.pi_value = None
        self.pi_func_optim = None
        self.pi_value_optim = None

        self.dummy_prior_func = None
        self.prior_func = None
        self.prior_value = None
        self.prior_func_optim = None
        self.prior_value_optim = None

        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        self.writer = SummaryWriter(self.args.log_dir)

    def sample_option(self, episode_id, init_state):
        with torch.no_grad():
            if (episode_id > self.args.keep_prior_iters) and (self.prior_func is not None):
                init_state = torch.tensor([init_state], device=self.args.device, dtype=torch.float32)
                c, _, _ = self.prior_func.forward(init_state)
            else:
                c = self.dummy_prior_func.sample((1,))
                c = c.to(self.args.device)
            c_onehot = F.one_hot(c, self.args.code_dim).float()

        return c, c_onehot # (1,), (1, code_dim)

    def sample_action(self, state, option):
        with torch.no_grad():
            state = torch.tensor([state], device=self.args.device, dtype=torch.float32)
            input = torch.cat([state, option], 1) # (1, obs_dim+code_dim)
            act, _, _ = self.pi_func.forward(input)
        return act # (1, ) or (1, act_dim)

    def sample_option_batch(self, episode_id, init_state):
        env_num = init_state.shape[0]
        with torch.no_grad():
            if (episode_id > self.args.keep_prior_iters) and (self.prior_func is not None):
                init_state = torch.tensor(init_state, device=self.args.device, dtype=torch.float32) # (env_num, obs_dim)
                c, _, _ = self.prior_func.forward(init_state) # (env, )
            else:
                c = self.dummy_prior_func.sample((env_num,))
                c = c.to(self.args.device)
            c_onehot = F.one_hot(c, self.args.code_dim).float()

        return c, c_onehot # (env,), (env, code_dim)

    def sample_action_batch(self, state, option):
        with torch.no_grad():
            state = torch.tensor(state, device=self.args.device, dtype=torch.float32)
            input = torch.cat([state, option], 1) # (env_num, obs_dim+code_dim)
            act, _, _ = self.pi_func.forward(input)
        return act # (env_num, ) or (env_num, act_dim)

    def train_with_env_rwd(self, episode_id, train_batch):
        # set up the data
        updated = train_batch.get_item("updated") # (bs, epi_len, 1)
        filled = train_batch.get_item("filled")  # (bs, epi_len, 1)
        horizons = train_batch.get_item("horizon")  # (bs, 1)
        states = train_batch.get_item("state")  # (bs, epi_len, obs_dim)
        options = train_batch.get_item("option")  # (bs, epi_len, 1)
        option_onehots = train_batch.get_item("option_onehot")  # (bs, epi_len, code_dim)
        acts = train_batch.get_item("action") # (bs, epi_len, 1) or (bs, epi_len, act_dim)
        rewards = train_batch.get_item("reward") # (bs, epi_len, 1)
        dones = train_batch.get_item("done") # (bs, epi_len, 1)
        next_states = train_batch.get_item("next_state")  # (bs, epi_len, obs_dim)
        bs = next_states.shape[0]
        epi_len = next_states.shape[1]
        log_info = {}

        # get target value
        if self.prior_func is None:
            next_options = self.dummy_prior_func.sample((bs,))
            next_options = next_options.to(self.args.device)
            next_options_onehot = F.one_hot(next_options, self.args.code_dim).float()  # (bs, code_dim)
        else:
            next_options, _, _ = self.prior_func.forward(next_states[:, -1]) # (bs, )
            next_options_onehot = F.one_hot(next_options.detach(), self.args.code_dim).float()  # (bs, code_dim)

        next_options_onehot = torch.cat([(option_onehots[:, 1:]).detach().clone(), next_options_onehot.unsqueeze(dim=1)], dim=1)
        next_pi_input = torch.cat([next_states, next_options_onehot], dim=-1).reshape(bs * epi_len, -1)  # (bs * epi_len, obs_dim + code_dim)

        target_v = self.pi_value.forward(next_pi_input)  # (bs * epi_len, 1)
        target_v = (target_v.reshape(bs, epi_len, 1) * filled * (1.0 - dones)).detach()  # (bs, epi_len, 1)
        final_v = get_final_state_value(self.args, target_v, horizons)
        # get return value
        ret = get_return_array(self.args, rewards, final_v, horizons, filled)
        log_info['return'] = ret.mean().item()

        # preparation
        pi_input = torch.cat([states, option_onehots], dim=-1).reshape(bs * epi_len, -1)  # (bs * epi_len, obs_dim + code_dim)
        if self.args.is_discrete:
            pi_func_a = acts.reshape(bs * epi_len)  # (bs*epi_len, )
        else:
            pi_func_a = acts.reshape(bs * epi_len, self.args.act_dim)  # (bs*epi_len, act_dim)

        # update the value
        for _ in range(self.args.value_iters):
            ret_pre = self.pi_value.forward(pi_input)  # (bs*epi_len, 1)
            pi_v_loss = (torch.square(ret_pre.reshape(bs, epi_len, 1) - ret) * filled).sum() / filled.sum()
            self.pi_value_optim.zero_grad()
            pi_v_loss.backward()
            self.pi_value_optim.step()
        log_info['pi_v_loss'] = pi_v_loss.item()

        # get advantage function
        curr_v = self.pi_value.forward(pi_input)
        curr_v = (curr_v.reshape(bs, epi_len, 1) * filled).detach()
        next_v = self.pi_value.forward(next_pi_input)
        next_v = (next_v.reshape(bs, epi_len, 1) * filled * (1.0 - dones)).detach()
        adv = get_advantage(self.args, rewards, curr_v, next_v, filled)  # (bs, epi_len, 1)
        if self.args.normalized:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        _, log_a, _ = self.pi_func.forward(pi_input, a=pi_func_a)  # (bs * epi_len, )
        log_a = log_a.view(bs, epi_len, 1).detach()  # (bs, epi_len, 1)

        for pi_iter in range(self.args.pi_iters):
            _, log_a_new, _ = self.pi_func.forward(pi_input, a=pi_func_a)  # (bs * epi_len, )
            ratio = torch.exp(log_a_new.view(bs, epi_len, 1) - log_a)  # log_a_old
            clip_adv = torch.clamp(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * adv
            pi_loss = -(torch.min(ratio * adv, clip_adv)).mean()
            # pi_loss = - (torch.min(ratio * adv, clip_adv) * filled).sum() / filled.sum()  # TODO: try not with the filled
            approx_kl = (log_a - log_a_new.view(bs, epi_len, 1)).mean().item()
            if approx_kl > 1.5 * self.args.target_kl:
                break

            self.pi_func_optim.zero_grad()
            pi_loss.backward()
            self.pi_func_optim.step()

        log_info['pi_loss'] = pi_loss.item()
        log_info['pi_iters'] = pi_iter

        if (self.prior_func is not None) and (self.prior_value is not None):
            # start to update the prior
            # get target value
            prior_target_v = self.prior_value.forward(next_states.reshape(bs * epi_len, -1))
            prior_target_v = (prior_target_v.reshape(bs, epi_len, 1) * filled * (1.0 - dones)).detach()  # (bs, epi_len, 1)
            prior_final_v = get_final_state_value(self.args, prior_target_v, horizons)
            # get return value
            prior_ret = get_return_array(self.args, rewards, prior_final_v, horizons, filled)
            log_info['prior_return'] = prior_ret.mean().item()

            # update the value function for the prior network
            for _ in range(self.args.value_iters):
                prior_ret_pre = self.prior_value.forward(states.view(bs * epi_len, -1))  # (bs * epi_len, 1)
                # prior_v_loss = (torch.square(prior_ret_pre.view(bs, epi_len, 1) - prior_ret) * filled).sum() / filled.sum()
                prior_v_loss = (torch.square(prior_ret_pre.view(bs, epi_len, 1) - prior_ret) * updated).sum() / updated.sum()  # TODO: use this line
                self.prior_value_optim.zero_grad()
                prior_v_loss.backward()
                self.prior_value_optim.step()
            log_info['prior_v_loss'] = prior_v_loss.item()

            # since for most of states, their value is not updated accordingly, we use the old way to calculate advantage, TODO
            prior_base = self.prior_value.forward(states.view(bs * epi_len, -1))  # (bs * epi_len, 1)
            prior_adv = (prior_ret - prior_base.view(bs, epi_len, 1)).detach()  # (bs, epi_len, 1)

            _, log_c, _ = self.prior_func.forward(states.view(bs * epi_len, -1), code_gt=options.squeeze(-1).view(bs * epi_len))  # (bs * epi_len, )
            log_c = log_c.view(bs, epi_len, 1).detach()

            for pi_iter in range(self.args.pi_iters):
                _, log_c_new, _ = self.prior_func.forward(states.view(bs * epi_len, -1), code_gt=options.squeeze(-1).view(bs * epi_len))  # (bs * epi_len, )
                ratio = torch.exp(log_c_new.view(bs, epi_len, 1) - log_c)
                clip_adv = torch.clamp(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * prior_adv
                prior_loss = - (torch.min(ratio * prior_adv, clip_adv) * updated).sum() / updated.sum()

                approx_kl = (log_c - log_c_new.view(bs, epi_len, 1)).mean().item()
                if approx_kl > 1.5 * self.args.target_kl:
                    break

                self.prior_func_optim.zero_grad()
                prior_loss.backward()
                self.prior_func_optim.step()
            log_info['prior_loss'] = prior_loss.item()
            log_info['prior_iters'] = pi_iter


        # log to tensorboard
        write_summary(self.writer, info=log_info, step=episode_id)
        print("Training info: ", log_info)

