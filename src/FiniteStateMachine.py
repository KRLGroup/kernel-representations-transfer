import random
import tempfile
from graphviz import Source
import sys
from pythomata import SimpleDFA
from pythomata import SymbolicAutomaton
# from flloat.parser.ltlf import LTLfParser
from ltlf2dfa.parser.ltlf import LTLfParser
import numpy as np
from itertools import chain, product
from DeepAutoma import DeepDFA, device

import torch


def dot2dfa(dot):
    # with tempfile.TemporaryFile() as file1:
    #     print(dot)
    #     file1.write(dot.encode())
    #     Lines = file1.readlines()
    #     print(f'lines: {Lines}')

    # with open('tmp.dot', 'w') as file1:
    #     file1.write(dot)

    # with open('tmp.dot', 'r') as file1:
    #     Lines = file1.readlines()
    #     # print(f'lines: {Lines}')

    Lines = dot.split('\n')
    # print(f'lines: {Lines}')

    count = 0
    states = set()

    for line in Lines:
        count += 1
        if count >=11:
            # print(f'line<6: {line}')
            if line.strip()[0] == '}':
                break
            action = line.strip().split('"')[1]
            states.add(line.strip().split(" ")[0])
        else:
            # print(f'line>6: {line}')
            if "doublecircle" in line.strip():
                final_states = line.strip().split(';')[1:-1]

    automaton = SymbolicAutomaton()
    state_dict = dict()
    state_dict['1'] = 0
    for state in states:
        if state == '1':
            continue
        state_dict[state] = automaton.create_state()


    final_state_list = []
    for state in final_states:
        state = int(state)
        state = str(state)
        final_state_list.append(state)

    for state in final_state_list:
        automaton.set_accepting_state(state_dict[state],True)


    count = 0
    for line in Lines:
        count += 1
        if count >=11:
            if line.strip()[0] == '}':
                break
            action = line.strip().split('"')[1]
            init_state = line.strip().split(" ")[0]
            final_state =  line.strip().split(" ")[2]
            automaton.add_transition((state_dict[init_state],action,state_dict[final_state]))


    automaton.set_initial_state(state_dict['1'])

    return automaton


class LTL2DFAError(ValueError):
    pass


class DFA:

    def __init__(self, arg1, arg2, arg3, dictionary_symbols = None):
        if dictionary_symbols == None:
            self.dictionary_symbols = list(range(self.num_of_symbols))
        else:
            self.dictionary_symbols = dictionary_symbols
        if isinstance(arg1, str):
            self.init_from_ltl(arg1, arg2, arg3, dictionary_symbols)
        elif isinstance(arg1, int):
            self.random_init(arg1, arg2)
        elif isinstance(arg1, dict):
            self.init_from_transacc(arg1, arg2)
        else:
            raise Exception("Uncorrect type for the argument initializing th DFA: {}".format(type(arg1)))


    def init_from_ltl(self, ltl_formula, num_symbols, formula_name, dictionary_symbols):

        #From LTL to DFA
        #   parser = LTLfParser()
        #   ltl_formula_parsed = parser(ltl_formula)
        #   dfa = ltl_formula_parsed.to_automaton()
        #   # print the automaton
        #   graph = dfa.to_graphviz()
        #   #graph.render("symbolicDFAs/"+formula_name)

        #   print(f'formula: {ltl_formula}')

        parser = LTLfParser()
        ast = parser(ltl_formula)
        dot = ast.to_dfa()

        try:
            dfa = dot2dfa(dot)
        except Exception as e:
            print(f'dfa conversion failed ({type(e)}), formula was {ltl_formula}, dot was: {dot}')
            raise LTL2DFAError(f'failed to convert formula to DFA: {ltl_formula}') from e

        dfa = dot2dfa(dot)

        # TODO dfa must be a pythomata dfa at this point (like before)
        #From symbolic DFA to simple DFA
        # print(dfa.__dict__)
        self.alphabet = dictionary_symbols
        self.transitions = self.reduce_dfa(dfa)
        # print(self.transitions)
        self.num_of_states = len(self.transitions)
        self.acceptance = []
        for s in range(self.num_of_states):
            if s in dfa._final_states:
                self.acceptance.append(True)
            else:
                self.acceptance.append(False)
        #print(self.acceptance)

        #Complete the transition function with the symbols of the environment that ARE NOT in the formula
        self.num_of_symbols = len(dictionary_symbols)
        self.alphabet = []
        for a in range(self.num_of_symbols):
            self.alphabet.append( a )
        if len(self.transitions[0]) < self.num_of_symbols:
            for s in range(self.num_of_states):
                for sym in  self.alphabet:
                    if sym not in self.transitions[s].keys():
                        self.transitions[s][sym] = s
        #print("Complete transition function")
        #print(self.transitions)
        #self.write_dot_file("simpleDFAs/{}.dot".format(formula_name))

    def reduce_dfa(self, pythomata_dfa):
        dfa = pythomata_dfa

        admissible_transitions = []
        for true_sym in self.alphabet:
            trans = {}
            for i, sym in enumerate(self.alphabet):
                trans[sym] = False
            trans[true_sym] = True
            admissible_transitions.append(trans)

        red_trans_funct = {}
        for s0 in dfa._states:
            red_trans_funct[s0] = {}
            transitions_from_s0 = dfa._transition_function[s0]
            for key in transitions_from_s0:
                label = transitions_from_s0[key]
                for sym, at in enumerate(admissible_transitions):
                    if label.subs(at):
                        red_trans_funct[s0][sym] = key

        return red_trans_funct

    def init_from_transacc(self, trans, acc):
        self.num_of_states = len(acc)
        self.num_of_symbols = len(trans[0])
        self.transitions = trans
        self.acceptance = acc

        self.alphabet = []
        for a in range(self.num_of_symbols):
            self.alphabet.append( a )

    def random_init(self, numb_of_states, numb_of_symbols):
        #print(f'num of states: {numb_of_states}')
        self.num_of_states = numb_of_states
        self.num_of_symbols = numb_of_symbols
        transitions= {}
        acceptance = []
        for s in range(numb_of_states):
            trans_from_s = {}
            #Each state is equiprobably set to be accepting or rejecting
            acceptance.append(bool(random.randrange(2)))
            #evenly choose another state from [i + 1; N ] and adds a random-labeled transition
            if s < numb_of_states - 1:
                s_prime = random.randrange(s + 1 , numb_of_states)
                a_start = random.randrange(numb_of_symbols)

                trans_from_s[a_start] = s_prime
            else:
                a_start = None
            for a in range(numb_of_symbols):
                #a = str(a)
                if a != a_start:
                    trans_from_s[a] = random.randrange(numb_of_states)
            transitions[s] = trans_from_s.copy()

        self.transitions = transitions
        self.acceptance = acceptance
        self.alphabet = ""
        for a in range(numb_of_symbols):
            self.alphabet += str(a)

    def accepts(self, string):
        if string == '':
            return self.acceptance[0]
        return self.accepts_from_state(0, string)

    def accepts_from_state(self, state,string):
        assert string != ''

        a = string[0]
        next_state = self.transitions[state][a]

        if len(string) == 1:
            return self.acceptance[next_state]

        return self.accepts_from_state(next_state, string[1:])

    def to_pythomata(self):
        trans = self.transitions
        acc = self.acceptance
        #print("acceptance:", acc)
        accepting_states = set()
        for i in range(len(acc)):
            if acc[i]:
                accepting_states.add(i)

        automaton = SimpleDFA.from_transitions(0, accepting_states, trans)

        return automaton

    def write_dot_file(self, file_name):
        with open(file_name, "w") as f:
            f.write(
                "digraph MONA_DFA {\nrankdir = LR;\ncenter = true;\nsize = \"7.5,10.5\";\nedge [fontname = Courier];\nnode [height = .5, width = .5];\nnode [shape = doublecircle];")
            for i, rew in enumerate(self.acceptance):
                    if rew:
                        f.write(str(i) + ";")
            f.write("\nnode [shape = circle]; 0;\ninit [shape = plaintext, label = \"\"];\ninit -> 0;\n")

            for s in range(self.num_of_states):
                for a in range(self.num_of_symbols):
                    s_prime = self.transitions[s][a]
                    f.write("{} -> {} [label=\"{}\"];\n".format(s, s_prime, self.dictionary_symbols[a]))
            f.write("}\n")

        s = Source.from_file(file_name)
        s.view()

    def calculate_state_representation_multiple_references_OLD(self, list_of_refs, max_trace_len, all_traces):
        #print(f'num of ref DFAs: {len(list_of_refs)}')
        states_repr = np.zeros((self.num_of_states, len(list_of_refs)))
        for i, ref in enumerate(list_of_refs):
            if isinstance(ref, DFA):
                states_repr[:,i] = self.calculate_state_representation_one_reference(ref.transitions, ref.acceptance, max_trace_len, all_traces)
            else:
                states_repr[:,i] = self.calculate_state_representation_one_reference(ref[0], ref[1], max_trace_len, all_traces)
        return states_repr

    def calculate_state_representation_one_reference_OLD(self, transitions, acceptance, maximum_len=None, all_traces=None):

        states_representation = np.zeros((self.num_of_states))
        #print(f'maximum_len: {maximum_len}')
        #print(self.alphabet)
        # all_traces = product(self.alphabet, repeat=maximum_len)
        if all_traces is None:
            assert maximum_len is not None
            print(f'No trace set specified, using all traces with given max len')
            all_traces = product(range(len(self.alphabet)), repeat=maximum_len)
        num_traces = 0

        for trace in all_traces:
            #print(trace)
            curr_states = list(range(self.num_of_states))
            curr_state_ref = 0
            for sym in trace:
                #simulate the trace on the reference DFA
                curr_state_ref = transitions[curr_state_ref][sym]
                curr_acc_ref = acceptance[curr_state_ref]
                num_traces +=1
                for state_id in range(self.num_of_states):
                    curr_states[state_id] = self.transitions[curr_states[state_id]][sym]
                    acc = self.acceptance[curr_states[state_id]]
                    if acc == curr_acc_ref:
                        states_representation[state_id] += 1

        #print(" 1:",states_representation)
        #print(" 2:",num_traces)
        return states_representation / num_traces


    def calculate_state_representation_multiple_references(self, list_of_refs, max_trace_len, all_traces):

        states_repr = np.zeros((self.num_of_states, len(list_of_refs)))
        for i, ref in enumerate(list_of_refs):
            if isinstance(ref, DFA):
                states_repr[:,i] = self.calculate_state_representation_one_reference_DEEP(ref.transitions, ref.acceptance, max_trace_len, all_traces)
            else:
                states_repr[:,i] = self.calculate_state_representation_one_reference_DEEP(ref[0], ref[1], max_trace_len, all_traces)
        return states_repr

    def calculate_state_representation_one_reference_DEEP(self, transitions, acceptance, maximum_len=None, all_traces=None):
        def list_of_tensors(list_of_tuples):
            # Create a dictionary to store tuples with the same length
            tuples_dict = {}

            # Iterate over the list of tuples
            for tpl in list_of_tuples:
                length = len(tpl)
                if length not in tuples_dict:
                    tuples_dict[length] = []
                tuples_dict[length].append(tpl)

            # Convert lists of tuples into tensors
            tensor_list = [torch.tensor(tuples, dtype=torch.int64) for tuples in tuples_dict.values()]

            return tensor_list

        final_states = [index for index, value in enumerate(acceptance) if value == True]
        num_states_ref = len(transitions)
        num_sym_ref = len(transitions[0])
        deepdfa = DeepDFA( num_sym_ref, num_states_ref, 2).to(device)
        deepdfa.initFromDfa(transitions, final_states)

        ego_dfa = DeepDFA(self.num_of_symbols, self.num_of_states, 2).to(device)
        final_states = [index for index, value in enumerate(self.acceptance) if value == True]
        ego_dfa.initFromDfa(self.transitions, final_states)

        states_representation = torch.zeros((self.num_of_states))

        if all_traces is None:
            assert maximum_len is not None
            # print(f'No trace set specified, using all traces with given max len')
            all_traces = product(range(len(self.alphabet)), repeat=maximum_len)

        all_traces = list(all_traces)
        traces_tensor = list_of_tensors(all_traces)

        for state in range(self.num_of_states):
            score = 0
            total = 0
            for batch in traces_tensor:
                batch = batch.to(device)
                initial_state = torch.zeros((batch.size()[0],self.num_of_states)).to(device)
                initial_state[:,state] = 1.0
                accept_ego = ego_dfa(batch, current_state= initial_state)
                accept_ref = deepdfa(batch)
                accept_ego = accept_ego.argmax(2)
                accept_ref = accept_ref.argmax(2)
                eq = (accept_ego == accept_ref).int()
                score += eq.sum()
                total += eq.numel()
            states_representation[state] = score / total
        return states_representation

    def to_arrays(self):
        transitions = np.zeros((self.num_of_states, self.num_of_symbols), dtype=int)
        for s in range(self.num_of_states):
            for sym in range(self.num_of_symbols):
                transitions[s][sym] = self.transitions[s][sym]
        acceptance = np.zeros((self.num_of_states), dtype=bool)
        for s in range(self.num_of_states):
            if self.acceptance[s]:
                acceptance[s] = True
        return transitions, acceptance


class MooreMachine(DFA):
    def __init__(self, arg1, arg2, arg3, reward = "acceptance", dictionary_symbols = None):
        super().__init__(arg1, arg2, arg3, dictionary_symbols)
        self.rewards = [100 for _ in range(self.num_of_states)]
        self.calculate_absorbing_states()
        if reward == "distance":
            for s in range(self.num_of_states):
                if self.acceptance[s]:
                    self.rewards[s] = 0
            #print(self.rewards)
            old_rew = self.rewards.copy()
            termination = False
            while not termination:
                termination = True
                for s in range(self.num_of_states):
                    if not self.acceptance[s]:
                        next = [ self.rewards[self.transitions[s][sym]] for sym in self.alphabet if self.transitions[s][sym] != s]
                        if len(next) > 0:
                            self.rewards[s] = 1 + min(next)

                termination = (str(self.rewards) == str(old_rew))
                old_rew = self.rewards.copy()

            for i in range(len(self.rewards)):
                self.rewards[i] *= -1
            minimum = min([r for r in self.rewards if r != -100])
            for i,r in enumerate(self.rewards):
                if r != -100:
                    self.rewards[i] = (r - minimum)

            maximum = max(self.rewards )
            #max : 100 = rew : x
            #x = 100 * rew / max
            for i,r in enumerate(self.rewards):
                if r != -100:
                    self.rewards[i] = 100 * r/ maximum
            print("REWARDS:", self.rewards)
            #assert False
        elif reward == "acceptance":
            for s in range(self.num_of_states):
                if self.acceptance[s]:
                    self.rewards[s] = 1
                else:
                    self.rewards[s] = 0
        elif reward == "three_value_acceptance":
            for q in range(self.num_of_states):
                #neutral states
                if q not in self.absorbing_states:
                    self.rewards[q] = 0
                else:
                    #winning state
                    if self.acceptance[q]:
                        self.rewards[q] = 1
                    #failure state
                    else:
                        self.rewards[q] = -1
        else:
            raise Exception("Reward based on '{}' NOT IMPLEMENTED, choose between ['acceptance', 'three_value_acceptance', 'distance']".format(reward))

    def calculate_absorbing_states(self):
        self.absorbing_states = []
        for q in range(self.num_of_states):
            absorbing = True
            for s in self.transitions[q].keys():
                    absorbing = absorbing & (self.transitions[q][s] == q)
            if absorbing:
                self.absorbing_states.append(q)

    def process_trace(self, trace):
        return self.process_trace_from_state(trace, 0)

    def process_trace_from_state(self, trace, state):
        a = trace[0]
        next_state = self.transitions[state][a]

        if len(trace) == 1:
            return next_state, self.rewards[next_state]

        return self.process_trace_from_state(trace[1:], next_state)
    def similarity_3_values(self, other, max_step):
            #questa funziona ma non distingue formule che sono diverse su tracce su cui una delle due è 0
            D = set([Trace([p]) for p in self.alphabet])
            similarity = 0
            norm_term =0

            for time in range(1, max_step + 1):
                next_D = set()
                print("D: __________________", D)
                for trace in D:
                    t = trace.trace
                    q1, o1 = self.process_trace(t)
                    q2, o2 = other.process_trace(t)
                    print(trace)
                    print(o1)
                    print(o2)
                    if o1 == 0 or o2 == 0:
                        candidates = set()
                        del_cand_traces = 0
                        for sym in self.alphabet:
                            q1_prime = self.transitions[q1][sym]
                            q2_prime = other.transitions[q2][sym]
                            if q1 == q1_prime and q2 == q2_prime:
                                del_cand_traces += 1
                            else:
                                candidates.add(Trace(t + [sym]))
                        for c in candidates:
                            c.delay_trace(del_cand_traces)
                        print("candidates: ", candidates)
                        next_D = next_D.union(candidates)
                    elif o1 == o2:
                        s = pow(len(self.alphabet), max_step - time)
                        similarity += s
                        norm_term += s
                        print("+ ", s)
                        if trace.delay:
                            sum = 0
                            for i in range(max_step - len(t) - 1):
                                sum += pow(len(self.alphabet), i)
                            s = trace.delay * sum
                            similarity += s
                            norm_term += s
                            print("+ ", s)
                    else:
                        s = pow(len(self.alphabet), max_step - time)
                        similarity -= s
                        norm_term += s
                        if trace.delay:
                            sum = 0
                            for i in range(max_step - len(t) - 1):
                                sum = pow(len(self.alphabet), i)
                            s = trace.delay * sum
                            similarity -= s
                            norm_term += s

                D = next_D

            #return similarity / float(pow(len(self.alphabet), max_step))
            return similarity / norm_term

    def similarity(self, other, max_step):

            D = set([Trace([p]) for p in self.alphabet])
            similarity = 0
            #norm_term =0

            for time in range(1, max_step + 1):
                next_D = set()
                print("D: __________________", D)
                for trace in D:
                    t = trace.trace
                    q1, o1 = self.process_trace(t)
                    q2, o2 = other.process_trace(t)
                    print(trace)
                    print(o1)
                    print(o2)
                    if q1 not in self.absorbing_states or q2 not in other.absorbing_states:
                        candidates = set()
                        del_cand_traces = 0
                        for sym in self.alphabet:
                            q1_prime = self.transitions[q1][sym]
                            q2_prime = other.transitions[q2][sym]
                            if q1 == q1_prime and q2 == q2_prime:
                                del_cand_traces += 1
                                if o1 == o2:
                                    similarity += 1
                                    print("+ 1")
                                else:
                                    similarity -= 1
                                    print("- 1")
                            else:
                                candidates.add(Trace(t + [sym]))
                        for c in candidates:
                            c.delay_trace(del_cand_traces)
                        print("candidates: ", candidates)
                        next_D = next_D.union(candidates)
                    else:
                        if o1 == o2:
                            s = pow(len(self.alphabet), max_step - time)
                            similarity += s
                            #norm_term += s
                            print("+ ", s)
                            if trace.delay:
                                sum = 0
                                for i in range(max_step - len(t) ):
                                    sum += pow(len(self.alphabet), i)
                                s = trace.delay * sum
                                similarity += s
                                #norm_term += s
                                print("+ ", s)
                        else:
                            s = pow(len(self.alphabet), max_step - time)
                            similarity -= s
                            #norm_term += s
                            print("- ", s)

                            if trace.delay:
                                sum = 0
                                for i in range(max_step - len(t) ):
                                    sum += pow(len(self.alphabet), i)
                                s = trace.delay * sum
                                similarity -= s
                                #norm_term += s
                                print("- ", s)

                D = next_D

            return similarity / float(pow(len(self.alphabet), max_step))
            #return similarity / norm_term
