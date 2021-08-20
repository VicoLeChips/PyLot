"""
This python file contains the class FA and all useful functions 
Class :
    - FA : Class that allows to create a finite automaton object and to use 
        its different methods (see docstring FA)
        
Functions :
    - read_automation_from_file : Function that creates and returns a finite automaton 
        by reading a given text file
    - read_word : storing in memory a string of characters typed by the user on the keyboard.
    - create_transition_liste : Function that makes a transition list from the list given 
        in the .TXT file
    - merge_list : Simple function to group two lists
    - merge_list_into_strings : Simple function to group two lists into string
    - get_transition_list : Fonction to get all the transitions from a table
    - get_nb_states : Fonction that return the number of keys from a dictionary
"""

__authors__ = ("BABIN Victor",
               "DEWATRE Pierre",
               "DHEILLY Robin", 
               "GREGOIRE Aurélien"
               "LEFÈVRE Valentin")
__date__ = "02/04/2021"

# Library
import string
# Secure importation
try:
    # Data analysis library that allow us to create the transition table
    import pandas
except ImportError:
    pandas = None
    print('Please import the data analysis library "pandas" with the command : "python -m pip install pandas"' 
          'if you do not install it, you will NOT have the transition table but only a list of transitions.')


class FA:
    """
    A class to represent a finite automaton.
    Attributes
        ----------
        number_A : int 
            number of char in the alphabet of the fa
        number_Q : int
            number of state
        number_I : int
            number of initial states
        number_T : int
            number of terminal states
        number_E : int
            number of transition
        I : list of str
            list of all the initial states
        T : list of str
            list of all the terminal states
        E : list of triplet p,x,q
            list of all the transition triplets 
            ex [['1','a','1+2'], ['1','b','1']] 
        dico_E : dict
            this attributes allow us to see quicker all the transitions for one state
            ex literal {'state': {'symbol': [next_state, next_state]}}
            ex value {"0": {'a': ['1','0'], 'b': ['1']}}
            If i want the next states of 0 with the symbol a i just say : dico_E['0']['a'] 
    Public Methods
        ----------
        FA(number_A, number_Q, number_I, I, number_T, T, number_E, E):
            Constructor of a fa
        display_automaton()
            Print the initial, terminal states and the transition table of the current automaton
        is_an_asynchronous_automaton():
            Return a bool to know if there are epsilon transitions in the current automaton
        determinization_and_completion_of_asynchronous_automaton()
            Return a new deterministe and complete automaton create from the current automaton
        determinization_and_completion_of_synchronous_automaton():
            Same but for a synchronous automaton
        is_deterministic():
            Return a bool to know if there is just one initial state
            and no more than one transition per symbol in the current automaton
        is_complete():
            Return a bool to know if there are transition for each symbols in each state
        completion():
            Return a complete automaton create from the current automaton
        minimization():
            Return a minimize automaton create from the current automaton
        recognize_word(word):
            Return a bool to know if the word is recognize by the current automaton
        complementary_automaton():
            Return the complementary automaton of the current automaton
        standard_automaton():
            Return a stadard automaton create from the current automaton
    Private Methods
        ----------
        table():
            Return a str transition table created from the current automaton
        _E_to_dico():
            Return a dictionnary of transition
        _create_dico()
            Return a new dico for the determinize automaton
        _states_to_study(dico):
            Return the list of the states to study
        _add_line_to_determinization_table()
        """

    def __init__(self, number_A, number_Q, number_I, I, number_T, T, number_E, E):
        """
        Constructs all the necessary attributes for the finite automaton object.
        Parameters
        ----------
        number_A : int 
            number of char in the alphabet of the fa
        number_Q : int
            number of state
        number_I : int
            number of initial states
        number_T : int
            number of terminal states
        number_E : int
            number of transition
        I : list of str
            list of all the initial states
        T : list of str
            list of all the terminal states
        E : list of triplet p,x,q
            list of all the transition triplets 
            ex : [['1','a','1+2'], ['1','b','1']] 
        dico_E : dictionary
            this attributes allow us to see quicker all the transitions for one state
            ex literal : {'state': {'symbol': [next_state, next_state]}}
            ex value : {"0": {'a': ['1','0'], 'b': ['1']}}
            If i want the next states of 0 with the symbol a i just say : dico_E['0']['a']
        """   
        self.number_A = number_A
        self.number_Q = number_Q
        self.number_I = number_I
        self.I = I
        self.number_T = number_T
        self.T = T
        self.number_E = number_E
        self.E = E
        self.dico_E = self._E_to_dico()

    def display_automaton (self):
        """
        Displaying the FA stored in memory, with an explicit indication:
            - of the initial state(s);
            - of the terminal state(s) ;
            - of the transition table.
        Returns :
            void
        """
        print("\nInitial state(s) :", self.I)
        print("Final state(s)  : ",self.T)
        # Secure importation
        if (pandas == None):
            print("\nList of all the transitions (because you do not have the pandas library) : ")
            # if we have no imporations, we just print the list of transition
            for transition in self.E :
                print(transition)
        else:
            print("\nTransitions table : ")
            print(self._table())

    def _table(self):
        """
        Generates a transition table of the given FA.
        Returns:
            table : A string transition table of the FA.
        """
        table: dict = {}
        # We go through our dico
        for state, transitions in self.dico_E.items():
            # Add arrow to know the initial and final state
            if state in self.I and state in self.T:
                state = "←→" + state
            elif state in self.I:
                state = "→" + state
            elif state in self.T:
                state = "←" + state

            # Now we fill the row (so the next state depending on the char)
            row: dict = {}

            for input_symbol, next_states in transitions.items():
                for next_state in next_states:
                    if input_symbol in row.keys():
                        if next_state in self.T:
                            for i in next_state:
                                #row[input_symbol] += ", " + "←" + i
                                row[input_symbol] += ", " + i
                        else:
                            for i in next_state:
                                row[input_symbol] += ", " + i
                    else:
                        if next_state in self.T:
                            #row[input_symbol] = "←" + next_state
                            row[input_symbol] = next_state
                        else:
                            row[input_symbol] = next_state
            table[state] = row
        table = pandas.DataFrame.from_dict(table).T
        # Replace the null indicator "NaN" by nothing
        return table.to_string().replace("NaN", "   ")

    def _E_to_dico(self):
        """ 
        Generates a dictionnary of the transitions of the current finite automata.
        This dictionnary allow us to see quicker all the transitions for one state
        Ex literal : {'state': {'symbol': [next_state, next_state]}}
        Ex value : {"0": {'a': ['1','0'], 'b': ['1']}}
        If i want the next states of 0 with the symbol a i just say : dico_E['0']['a'] 
        Returns : 
            dictionnary : list the transition by starting states
        """
        dico = {}
        # We go through all our transition (reminder : E is like [['0', 'a', '1']['0', 'b', '0']['1', 'a', '1']])
        for transition in self.E :
            # If the stating state already in our dictionnary
            if (transition[0] in dico.keys()):
                # If the symbol already has an associate next state
                if (transition[1] in dico[transition[0]].keys()):
                    # We append this state
                    dico[transition[0]][transition[1]].append(transition[2])
                else:
                    # Else we add the couple symbol next state under the form : {'symbol': ['next state']}
                    dico[transition[0]][transition[1]] = [transition[2]]
            else:
                # Else we add this transition on the dictionnary under the form : {..., 'stating state': {'symbol': ['next state']}}
                dico[transition[0]] = {transition[1]: [transition[2]]}
            if (transition[2] not in dico.keys()):
                dico[transition[2]] = {}
        return dico

    def is_an_asynchronous_automaton(self):
        """
        Allow us to know if the current automaton has epsilon transitions
        Returns :
            Bool : True if the automaton is asynchronous
        """
        # Go through our transitions
        for transition in self.E:
            # If we have an epsilon transitions
            if ('*' in transition):
                # Explains why it is asynchronous
                print("\033[1m➞ This is an asynchronous automaton because of the transition\033[0m", transition)
                return True
        print("\033[1m➞ This is not an asynchronous automaton\033[0m")
        return False

    def determinization_and_completion_of_asynchronous_automaton(self):
        """
        !!!! A finir
        """
        synchronousFA = self.epsilon_closure()
        synchronousFA.display_automaton()
        determinizedFA = synchronousFA.determinization_and_completion_of_synchronous_automaton()
        return determinizedFA
        
    def determinization_and_completion_of_synchronous_automaton(self):
        table, final_states = self._create_dico()

        merge_list_into_strings(table)
        list_transitions = get_transition_list(table)
        newFA = FA(self.number_A, get_nb_states(table), 1, [list(table.keys())[0]], len(final_states), final_states, len(list_transitions), list_transitions)
        return newFA.completion()

    def _create_dico(self):
        """
        """
        dico = {}
        final_states = self.T
        studied_states = self.I
        while (studied_states):
            for i in studied_states:
                if i in self.T:
                    final_states.append("+".join(studied_states))
            given_transitions = self._add_line_to_determinization_table(studied_states)
            dico["+".join(studied_states)] = given_transitions
            studied_states = self._states_to_study(dico)
        return (dico, final_states)

    def _states_to_study(self, dico):
        for i in dico.keys():
            for j in dico[i].keys():
                if ((len(dico[i][j]) != 0) and ("+".join(dico[i][j]) not in dico.keys())):
                    return dico[i][j]
        return []

    def _add_line_to_determinization_table(self, studied_states):
        given_transitions = {}
        for i in range(0, self.number_A):
            j = chr(i+97)
            given_transitions[j] = merge_list(list(map(self._for_map_get_index, studied_states, [j]*len(studied_states))))
        return (given_transitions)

    def _for_map_get_index(self, index, j):
        if (index in self.dico_E.keys()):
            if (j in self.dico_E[index]):
                return self.dico_E[index][j]
            else:
                return []
        else:
            return []

    def epsilon_closure(self):
        newDico = self.dico_E.copy()
        I = self.I
        T = self.T
        for state in newDico.keys():
            while ("*" in newDico[state].keys() and len(newDico[state]['*']) != 0):
                for i in newDico[state]["*"]:
                    if (state in I and i not in I):
                        I.append(i)
                    if (state in T and i not in T):
                        T.append(i)
                    if (i in newDico.keys()):
                        for j in newDico[i].keys():
                            for k in newDico[i][j]:
                                if (j in newDico[state].keys()):
                                    newDico[state][j].append(k)
                                else:
                                    newDico[state][j] = [k]
                    newDico[state]["*"].remove(i)

        merge_list_into_strings(newDico)
        list_transitions = get_transition_list(newDico)
        return FA(self.number_A, get_nb_states(newDico), len(I), I, len(T), T, len(list_transitions), list_transitions)
        

    def is_deterministic (self):
        """
        Allow us to know if there is just one initial state
        and no more than one transition per symbol in the current automaton
        Returns :
            bool : True if the current automaton is deterministic
        """
        if (self.number_I == 1 and not self._more_than_one_transition_per_symbol()):
            print("\033[1m➞ This is a deterministic automaton\033[0m")
            return True
        else:
            print("\033[1m➞ This is not a deterministic automaton\033[0m")
            return False

    def _more_than_one_transition_per_symbol(self):
        """
        Allow us to know if there are more than one transition per symbol in the current automaton
        Returns :
            bool : True if there are more than one transition per symbol
        """
        # To optimize
        for transitions in self.dico_E.values():
            for transition in transitions.values():
                # If there are more than one
                if len(transition) >= 2:
                    return True
        return False

    def is_complete(self):
        """
        Allow us to know if there are transition for each symbols in each state
        Returns :
            bool : True if the current automaton is complete
        """
        # Go through our dictionnary of transitions
        for state, transitions in self.dico_E.items():
            # If the number of symbol is less than the number of symbol in the automaton alphabet
            if len(transitions.keys()) != self.number_A:
                print("\033[1m➞ This is not a complete automaton because there are no transition for each symbols in the state : \033[0m", state)
                return False
        print("\033[1m➞ This is a complete automaton\033[0m")
        return True

    def completion(self):
        """
        Allow us to create a complete automaton create from the current automaton
        Returns :
            FA : complete automaton create from the current automaton
        """
        # We create a P state where will go all the new transition
        number_Q = self.number_Q + 1
        E = self.E
        for states, transitions in self.dico_E.items():
            for symbol_value in range(self.number_A):
                # If there is no transition for a symbol
                if chr(97+symbol_value) not in transitions.keys():
                    # Create a transition to the 'P' state
                    E.append([states, chr(97+symbol_value), 'P'])
        # Finally we create the P state transition that go into itself
        for symbol_value in range(self.number_A) :
            E.append(['P', chr(97+symbol_value), 'P'])
        # We create the new automaton from the information
        fa = FA(self.number_A, number_Q, self.number_I, self.I, self.number_T,
                self.T, len(E), E)
        return fa

    def minimization(self):
        """
        Allow us to create a minimize automaton create from the current automaton
        Returns: FA
        """ 
        not_T = []
        for state in self.dico_E.keys():
            if state not in self.T:
                not_T.append(state)
        
        partition = [(self.T).copy(), not_T]
        splited = True
        while (splited):
            splited = False
            for part in partition:
                if len(part) > 1:
                    if (self._split_part(part, splited, partition)):
                        splited = True
        
        I = prevent_doublons(list(map(lambda state: search_for_state(state, partition), self.I)))
        T = prevent_doublons(list(map(lambda state: search_for_state(state, partition), self.T)))
        E = []
        
        for state in partition:
            if (len(state) != 0):
                if (state[0] in self.dico_E):
                    for symbol, arrival in self.dico_E[state[0]].items():
                        E.append([search_for_state(state[0], partition), symbol, search_for_state(arrival[0], partition)])

        #return FA(self.number_A, len(partition), len(I), I, len(T), T, len(E), E)
        return FA(self.number_A, len(partition), len(I), I, len(T), T, len(E), E)._remove_single_elements()
    
    def _remove_single_elements(self):
        """
        This method is used to complete the minimization function: it will remove all the elements that could not be accessed
        Returns: FA
        """
        statesToRemove = []
        for state in self.dico_E.keys():
            toRemove = True
            for i in self.dico_E.keys():
                for j in self.dico_E[i].keys():
                    if (state != i and state == self.dico_E[i][j][0]):
                        toRemove = False
            if (toRemove and (state not in self.I)):
                statesToRemove.append(state)
        for state in statesToRemove:
            self.dico_E.pop(state)
        E = get_transition_list(self.dico_E)
        I = self.I
        T = list(set(self.dico_E.keys()).intersection(self.T))

        return FA(self.number_A, get_nb_states(self.dico_E), len(I), I, len(T), T, len(E), E)
    
    def _split_part(self, part, splited, partition):
        """
        docstring
        """
        splited = False
        i = 0
        continueSearching = True
        while ((i < len(part) -1) and continueSearching):
            new_part = []
            if self._are_separate_states(part[i], part[i+1], partition):
                continueSearching = False
                splited = True
                new_part.append(i)
                for j in range(0, len(part)):
                    if ((j != i) and (not self._are_separate_states(part[i], part[j], partition))):
                        new_part.append(j)
            if splited:
                new_part.sort()
                partition.insert(0, [])
                for i in range(0, len(new_part)):
                    partition[0].append(part.pop(new_part[i]-i))
                    
            i += 1
        return splited

    def _are_separate_states(self, p1, p2, partition):
        """
        docstring
        """
        list_q1 = []
        list_q2 = []
        for symbol_value in range(0, self.number_A):
            if (p1 in self.dico_E.keys()):
                list_q1.append(self.dico_E[p1][chr(97+symbol_value)][0])
            if (p2 in self.dico_E.keys()):
                list_q2.append(self.dico_E[p2][chr(97+symbol_value)][0])
        if (len(list_q1) > 0 and len(list_q2) > 0):
            for i in range(len(list_q1)):
                for part in partition:
                    if (list_q1[i] in part and list_q2[i] not in part) or (list_q1[i] not in part and list_q2[i] in part):
                        return True
        return False

    def read_word(self):
        """
        Storing in memory a string of characters typed by the user on the keyboard.
        Changed Parameters:
            str : the word typed by the user
        """
        alphabet = []
        for i in range (97, self.number_A + 97):
            alphabet.append(chr(i))
            print()
        valid = False
        while not valid:
            valid = True
            word = input("➞ Please enter a word in the alphabet : " + str(alphabet) + ": ")
            for char in word:
                if (char not in alphabet) and char != '*' and word != "end":
                    valid = False
        return word

    def recognize_word(self, word):
        """
        Allow us to know if the word is recognize by the current automaton
        Returns:
            bool : True if the word is recognize by the current automaton
        """
        recognized = True
        if word == "*":
            if self.I[0] in self.T:
                print("The empty word is recognized")
            else:
                print("The empty word is not recognized")
        else:
            current_state = self.I[0]
            for symbol in word :
                print(symbol)
                if (current_state in self.dico_E.keys()):
                    current_state = self.dico_E[current_state][symbol][0]
                else:
                    recognized = False
            if current_state not in self.T:
                recognized = False
            if (recognized):
                print("The word :", word, "is recognized")
            else:
                print("The word :", word, "is not recognized")

    def complementary_automaton(self):
        """
        Allow us to create the complementary automaton of the current automaton
        Returns:
            FA : complementary automaton
        """
        not_T = []
        for state in self.dico_E.keys():
            if state not in self.T:
                not_T.append(state)
        complementary = FA(self.number_A, self.number_Q, self.number_I, self.I, len(not_T), not_T, self.number_E, self.E)
        return complementary

    def standard_automaton(self):
        """
        Allow us to create the standard automaton of the current automaton
        Returns:
            FA : standard automaton
        """
        number_Q = self.number_Q + 1
        number_I = 1
        I = ['i']
        for state in self.I:
            if state in self.T:
                number_T = self.number_T + 1
                T = self.T
                T.append("i")
            else:
                number_T = self.number_T
                T = self.T
        number_E = self.number_E
        E = self.E
        for state in self.I:
            if (state in self.dico_E.keys()):
                for symbol in list(self.dico_E[state].keys()):
                    for j in self.dico_E[state][symbol]:
                        if [j] not in E:
                            E.append(['i', symbol, j])
                            number_E += 1
        standard = FA(self.number_A, number_Q, number_I, I, number_T, T, number_E, E)
        return standard


def read_automaton_from_file(filename):
    """
    This function will create a Finite Automate object from a file.
    It will open the file, read trough it in order to get all the informations,
    then call the constructor of the FA class to create and return the object.
    Parameters:
        str : filename to open
    Returns : 
        FA : the fa create frome the file
    """
    # We start by opening the file
    fileFA = open("Automata/"+filename, "r")
    # The 'rstrip' method is used to get rid of the '\n' character since it's part of the line that is readed by the 'readLine' method, but we don't want it in the result
    number_A = int(fileFA.readline().rstrip('\n'))
    number_Q = int(fileFA.readline().rstrip('\n'))
    # The 'split' method is used to create a list from the array, separating each elements by the ' '
    I = fileFA.readline().rstrip('\n').split(' ')
    number_I = int(I.pop(0))
    T = fileFA.readline().rstrip('\n').split(' ')
    number_T = int(T.pop(0))
    number_E = int(fileFA.readline().rstrip('\n'))
    E = []
    # Go through all the line and append the list
    for i in range(0, number_E):
        E.append(create_transition_list(list(fileFA.readline().rstrip('\n'))))
    # Sort the list by the starting state
    E.sort(key = lambda E: int(E[0]))
    fa = FA(number_A, number_Q, number_I, I, number_T, T, number_E, E)
    fileFA.close()
    return fa
  
def create_transition_list(readed_list):
    """
    Function that makes a transition list from the list given in the .TXT file
    Returns :
        list : list of one transition
    """
    new_list = [readed_list[0]]
    i = 1
    # Allow us the use the starting state and after the next state if there are greater than 9
    while readed_list[i].isnumeric():
        new_list[0] += readed_list[i]   
        i += 1
    new_list.append(readed_list[i])
    i += 1
    new_list.append(readed_list[i])
    i += 1
    while i < len(readed_list) :
        new_list[2] += readed_list[i]
        i += 1
    return new_list

def merge_list(to_merge):
    """
    Simple function to group two lists
    Returns :
        List : merge of both list
    """
    new_list = []
    for i in to_merge:
        for j in i:
            if (j not in new_list):
                new_list.append(j)
    return sorted(new_list)

def merge_list_into_strings(to_merge):
    """
    Simple function to group two lists into string
    Modified value :
        list : to_merge
    """
    for i in to_merge:
        for j in to_merge[i]:
            to_merge[i][j] = "+".join(to_merge[i][j])

def get_transition_list(table):
    """
    Fonction to get all the transitions from a table
    Returns:
        list : transitions
    """
    list_transitions = []
    for i in table:
        for j in table[i]:
            if (table[i][j]):
                if (isinstance(table[i][j], str)):
                    list_transitions.append([i, j, table[i][j]])
                elif (isinstance(table[i][j], list)):
                    list_transitions.append([i, j, table[i][j][0]])
    return list_transitions

def get_nb_states(dico):
    """
    Fonction to get the number of state of a new dico (not created from a fa)
    Return :
        int : number of starting state in the dico
    """
    return len(list(dico.keys()))

def search_for_state(state, partition):
    """
    This function returns the index (as a string) of an element in a partition.
    In other words, it simply returns the 
    """
    for i in range(0, len(partition)):
        for j in range(0, len(partition[i])):
            if partition[i][j] == state:
                return str(i)

def prevent_doublons(list_to_filter):
    """
    This fonction takes a list in parameters and returns the same list, but with each value appearing only once.
    Returns: list
    """
    newList = []
    for i in list_to_filter:
        if i not in newList:
            newList.append(i)
    return newList
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
