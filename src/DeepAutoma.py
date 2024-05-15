import torch
import torch.nn as nn

# if torch.cuda.is_available():
#     device = 'cuda:0'
# else:
#     device = 'cpu'
device = 'cpu'

print("Device:", device)

class DeepDFA(nn.Module):
    def __init__(self, numb_of_actions, numb_of_states, numb_of_output):
        super(DeepDFA, self).__init__()
        self.numb_of_actions = numb_of_actions
        self.alphabet = [str(i) for i in range(numb_of_actions)]
        self.numb_of_states = numb_of_states
        self.numb_of_outputs = numb_of_output
        self.output_values = torch.Tensor(list(range(numb_of_output)))
        self.trans_prob = torch.normal(0, 0.1, size=( numb_of_actions, numb_of_states, numb_of_states), requires_grad=False, device=device)
        self.fin_matrix = torch.normal(0, 0.1, size=( numb_of_states, numb_of_output), requires_grad=False, device=device)

    #input: sequence of actions (batch, length_seq)
    def forward(self, action_seq, current_state= None):
        batch_size = action_seq.size()[0]
        length_seq = action_seq.size()[1]
        rewards = torch.zeros((batch_size, length_seq, self.numb_of_outputs)).to(device)
        if current_state == None:
            s = torch.zeros((batch_size,self.numb_of_states)).to(device)
            #initial state is 0 for construction
            s[:,0] = 1.0
        else:
            s = current_state
        for i in range(length_seq):
            a = action_seq[:,i]
            s, r = self.step(s, a)
            rewards[:, i:i+1, :] = r.unsqueeze(1).expand(-1, 1, -1)
        return rewards

    def step(self,state, action):
        if type(action) == int:
            action= torch.IntTensor([action])

        selected_prob = torch.index_select(self.trans_prob, 0, action)
        next_state = torch.matmul(state.unsqueeze(dim=1), selected_prob)

        next_output = torch.matmul(next_state, self.fin_matrix)
        next_state = next_state.squeeze(1)
        next_output = next_output.squeeze(1)
        return next_state, next_output

    def initFromDfa(self, reduced_dfa, final_states):
        with torch.no_grad():
            #zeroing transition probabilities
            for a in range(self.numb_of_actions):
                for s1 in range(self.numb_of_states):
                    for s2 in range(self.numb_of_states):
                        self.trans_prob[a, s1, s2] = 0.0

            #zeroing output matrix
            for s in range(self.numb_of_states):
                for r in range(self.numb_of_outputs):
                    self.fin_matrix[s,r] = 0.0


        #set the transition probabilities as the one in the dfa
        for s in reduced_dfa:
            for a in reduced_dfa[s]:
                with torch.no_grad():
                    self.trans_prob[a, s, reduced_dfa[s][a]] = 1.0

        #set final matrix
        for s in range(len(reduced_dfa.keys())):
            if s in final_states:
                with torch.no_grad():
                    self.fin_matrix[s, 1] = 1.0
            else:
                with torch.no_grad():
                    self.fin_matrix[s, 0] = 1.0


