#########################  Bond Extraction Detail  ##############################################
# Python code to extract detail of Types of carbon in the given SMILE mainly for alkane
################################################################################################
# Note:Classification of carbon as primary , secondary, tertiary or quarternary is based on the number
# of carbon it is bonded to. This classifications only applies to satuaretd carbon(saturated hydrocarbon 
# are atom, in which are bonded together with single bonds. Unsaturated hydrocarbon have double bond 
# between carbons atoms, which are more reactive )
#################################################################################################


import sys
import os
import pandas as pd
import numpy as np
from rdkit import Chem as Chem
import collections
from search_fileNcreate import search_fileNcreate as SF
class Bond_Extraction:

    def check_ring(m):
        ri = m.GetRingInfo()
        return ri.NumRings()

    def Bond_Extract(Unique_fuel_smiles,curr_directory,file_name='Bond_detail.csv'):
        #Adding library 
        try:
                # '''
                # If  externally features are supplied given more prioritys
                # '''
                sys.path.append(curr_directory)
                from feature_selection import select_feature as Sel_feat
        except ImportError:
                from select_feature import select_feature as Sel_feat
        
        #columns
        columns = Sel_feat.bond_extraction_cols()
        Bond_detail_dataframe = pd.DataFrame(columns=columns)
        for i,item in enumerate(Unique_fuel_smiles):
            print('Unique fuel smiles : ', Unique_fuel_smiles[i])
            # Molfile object from Smiles
            if (Unique_fuel_smiles[i] == 'C'):
                #Applicable only for methane 
                C_data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0]
                C_data = pd.Series(C_data,index=columns)
                Bond_detail_dataframe=Bond_detail_dataframe.append(C_data,ignore_index=True)
                continue
            if (Unique_fuel_smiles[i] != 'C'):

                # file.write(str(Unique_fuel_smiles[i]))
                m = Chem.MolFromSmiles(Unique_fuel_smiles[i])  # obtaing Object

                # if input is none
                if (m is None):
                    print(str("\nWrong SMILES...\nCannot process species..."))
                    print(str("\n==========================================\n"))
                    sys.exit(1)

                # Checking weather Aromatic or not
                ri = m.GetRingInfo()
                n_rings = ri.NumRings()

                if (n_rings > 0):
                    print(str("\nThis is a cyclic species !!!\nProgram exiting..."))
                    print(str("\n==========================================\n"))
                    sys.exit(1)

                # Molfile block From Molfile object
                m_blk = Chem.MolToMolBlock(m)
                # print(m_blk)
                
                # file split by line array
                lines = m_blk.split('\n')

                # length of Smiles only Character
                Smiles_length = 0  # Initialize
                Smiles_char = []  # Smiles Character List
                # checking character is alphabat or not for processing of file
                for j,item in enumerate(Unique_fuel_smiles[i]):
                    if (Unique_fuel_smiles[i][j].isalpha()):
                        Smiles_length += 1
                        Smiles_char.append(Unique_fuel_smiles[i][j])

                ######################################################
                # Grabbing Bond data information
                Bond_data_end = 6+2*Smiles_length-3  # based on general MOL file ouptut format
                Bond_data_start = Smiles_length+4

                #Bond_data_columns = ['First_atom','Second_atom','Bond_type']

                Bond_data = pd.DataFrame(dtype='int32')
                # Capturing bond information only, seperated data by space
                for i in range(Bond_data_start, Bond_data_end, 1):
                    line_to_array = lines[i].split(" ")
                    # line to list by ignoring the empty entries
                    line_to_array = [x for x in line_to_array if x]
                    line_to_array = pd.Series(line_to_array[0:3])
                    Bond_data = Bond_data.append(line_to_array, ignore_index=True)

                # changing the DataFrame type
                for i in range(Bond_data.shape[1]):
                    Bond_data[i] = pd.to_numeric(Bond_data[i])

                # Note : List index starts from 0
                # First_atom chemical symbol list of the bond    C-O ; C is the first atom
                First_atoms = []
                for i in range(Bond_data.shape[0]):
                    First_atoms.append(Smiles_char[Bond_data.iloc[i, 0]-1])
                    # -1 as bond_data starts from 0 and Smiles_char index starts from 0

                # Second_atom chemical symbol list of the bond    C-O ; O is the first atom
                Second_atoms = []
                for i in range(Bond_data.shape[0]):
                    Second_atoms.append(Smiles_char[Bond_data.iloc[i, 1]-1])

                # Append list as column
                Bond_data[3] = First_atoms
                Bond_data[4] = Second_atoms
                # '''
                # #Bond Data Result
                # ['C', 'C', 'C', 'C', 'O', 'C', 'C']
                # first atom , second atom , no. of bonds , first atom symbol , second atom symbol
                # 0  1  2  3  4
                # 0  1  2  1  C  C
                # 1  2  3  1  C  C
                # 2  2  4  1  C  C
                # 3  4  5  1  C  C
                # 4  5  6  1  C  O
                # 5  5  7  1  C  C
                # 6  5  8  1  C  C
                # '''

                Improved_Bond_data = pd.DataFrame(Bond_data)
                for i in range(Bond_data.shape[0]):
                    if (Bond_data.iloc[i, 3] != 'C' or Bond_data.iloc[i, 4] != 'C'):
                        Improved_Bond_data.drop(
                            Improved_Bond_data.index[i], inplace=True)
                # '''
                # #Improved Data Result
                # ['C', 'C', 'C', 'C', 'O', 'C', 'C']
                # first atom , second atom , no. of bonds , first atom symbol , second atom symbol
                # 0  1  2  3  4
                # 0  1  2  1  C  C
                # 1  2  3  1  C  C
                # 2  2  4  1  C  C
                # 3  4  5  1  C  C
                # 5  5  7  1  C  C
                # 6  5  8  1  C  C
                # '''


                # DataFrame to Array of first two columns of Improved_Bond_data to find out bond type
                # Repetation of specific number suggestes waether it is
                # Primary secondary ....or quarternary carbon
                Bond_list_first_atom = Improved_Bond_data[0].tolist()
                Bond_list_second_atom = Improved_Bond_data[1].tolist()
                Bond_list = Bond_list_first_atom + Bond_list_second_atom

                ##################################################################
                # Returns the Dictionary number of connection of central atom with other Carbon atoms
                connection_list = collections.Counter(Bond_list)
                #### {2: 3, 5: 3, 4: 2, 1: 1, 3: 1, 7: 1, 8: 1}

                # Connection list into the dataframe
                connection_Series = pd.Series()
                # First column:Central_Carbon , Second column:Other_Carbon_Connection
                connection_data = pd.DataFrame()
                for i in sorted(connection_list):
                    connection_Series = [i, connection_list[i]]
                    connection_data = connection_data.append([connection_Series])


                # Handaling Data of Dictionary
                # for i in range(Smiles_length):
                #     #+1 as smiles legth starts from 0 index
                #     #where central atom index starts from zero as in dictionary
                #     connection_list[i+1]

                # Carbon type list based on its position in the SMILE
                carbon_type_list = []
                for i in range(connection_data.shape[0]):
                    if (connection_data.iloc[i, 1] == 1):
                        carbon_type_list.append('P')
                        continue
                    if (connection_data.iloc[i, 1] == 2):
                        carbon_type_list.append('S')
                        continue
                    if (connection_data.iloc[i, 1] == 3):
                        carbon_type_list.append('T')
                        continue
                    if (connection_data.iloc[i, 1] == 4):
                        carbon_type_list.append('Q')
                        continue
                    else:
                        carbon_type_list.append('NaN')

                # Adding column to connection_data at loation 2
                connection_data[2] = carbon_type_list

                # '''
                # central_atom no. based on smile , connection witBond_detail_dataframe
                # 0  4  2  S
                # 0  5  3  T
                # 0  7  1  P
                # 0  8  1  P
                # '''
                # Dictionary of first col of connection_data and carbon_type_list
                carbon_atom_posi = list(connection_data.iloc[:, 0])
                #Dictionary 
                carbon_posi_type_dict = dict(zip(carbon_atom_posi,carbon_type_list))
                # '''{1: 'P', 2: 'T', 3: 'P', 4: 'S', 5: 'T', 7: 'P', 8: 'P'}'''

                #Array of Smile atom number by it spostion or length (CCCC)=(1234)
                Smile_number = list(range(1,Smiles_length+1))

                #Smile based type of carbon list
                carbon_type =  []
                for i in range(Smiles_length):
                    if (Smile_number[i] in carbon_posi_type_dict):
                        carbon_type.append(carbon_posi_type_dict[i+1])  
                        #+1 as i starts from zero but dictionary entries starts from 1
                    else:
                        carbon_type.append("NaN")

                #Smile Based Dictionary ...Atom postion and its type 
                Smile_dict= dict(zip(Smile_number,carbon_type))
                # '''{1: 'P', 2: 'T', 3: 'P', 4: 'S', 5: 'T', 6: 'NaN', 7: 'P', 8: 'P'}'''
                Smiles_carbon_type = list(Smile_dict.values())  #Carbon type related to smile atom position 

                #Carbon Types
                Smiles_carbon_type_dict = collections.Counter(Smiles_carbon_type)
                #({'P': 5, 'Q': 1, 'S': 1, 'T': 1, 'NaN': 1})

                #To add list in the Bond_dataset
                carbon_type_array = []
                for i in range (5) :
                    if (i==0):
                        carbon_type_array.append(Smiles_carbon_type_dict.get('P'))
                    if (i==1):
                        carbon_type_array.append(Smiles_carbon_type_dict.get('S'))
                    if (i==2):
                        carbon_type_array.append(Smiles_carbon_type_dict.get('T'))
                    if (i==3):
                        carbon_type_array.append(Smiles_carbon_type_dict.get('Q'))
                    if (i==4):
                        carbon_type_array.append(Smiles_carbon_type_dict.get('NaN'))
                # file.write(str(carbon_type_array))
            
                
                #Using Smile_dict Dictionary. Find information of first and second atom 
                #of Bond_data Table for processing of  Bond detail 
                Bond_data_1statoms = Bond_data.iloc[:,0]
                Bond_data_2ndatoms = Bond_data.iloc[:,1]
                Bond_data_1statom_Ctype = []
                Bond_data_2ndatom_Ctype = []

                #Running Loop to generate type of carbon for Bond_data Dataframe
                for i in range(Bond_data.shape[0]):
                    Bond_data_1statom_Ctype.append(Smile_dict.get(Bond_data_1statoms[i]))
                    Bond_data_2ndatom_Ctype.append(Smile_dict.get(Bond_data_2ndatoms[i]))

                Bond_data[5] = Bond_data_1statom_Ctype
                Bond_data[6] = Bond_data_2ndatom_Ctype

                # file.write(str(Bond_data_1statom_Ctype))
                # file.write(str(Bond_data_2ndatom_Ctype))

                #Generate array to append into main table 
                #Bonds Count
                P_P = 0 #1
                P_S = 0 #2
                P_T = 0 #3
                P_Q = 0 #4
                S_S = 0 #5
                S_T = 0 #6 
                S_Q = 0 #7
                T_T = 0 #8
                T_Q = 0 #9
                Q_Q = 0 #10
                P_H = 0 #11
                S_H = 0 #12 
                T_H = 0 #13
                Q_H = 0 #Doesn't make sense

                #Bond Count of different atoms
                for i in range(Bond_data.shape[0]):
                    if(Bond_data_1statom_Ctype[i] == 'P' and Bond_data_2ndatom_Ctype[i] =='P' ):
                        P_P += 1
                        continue
                
                    if(Bond_data_1statom_Ctype[i] == 'P' and Bond_data_2ndatom_Ctype[i] =='S' ):
                        P_S += 1
                        continue
                    
                    if(Bond_data_1statom_Ctype[i] == 'S' and Bond_data_2ndatom_Ctype[i] =='P' ):
                        P_S += 1
                        continue
                    
                    if(Bond_data_1statom_Ctype[i] == 'P' and Bond_data_2ndatom_Ctype[i] =='T' ):
                        P_T += 1
                        continue
                    
                    if(Bond_data_1statom_Ctype[i] == 'T' and Bond_data_2ndatom_Ctype[i] =='P' ):
                        P_T += 1
                        continue
                    
                    if(Bond_data_1statom_Ctype[i] == 'P' and Bond_data_2ndatom_Ctype[i] =='Q' ):
                        P_Q += 1
                        continue
                    
                    if(Bond_data_1statom_Ctype[i] == 'Q' and Bond_data_2ndatom_Ctype[i] =='P' ):
                        P_Q += 1
                        continue
                                    
                    if(Bond_data_1statom_Ctype[i] == 'S' and Bond_data_2ndatom_Ctype[i] =='S' ):
                        S_S += 1
                        continue
                    
                    if(Bond_data_1statom_Ctype[i] == 'S' and Bond_data_2ndatom_Ctype[i] =='T' ):
                        S_T += 1
                        continue

                    if(Bond_data_1statom_Ctype[i] == 'T' and Bond_data_2ndatom_Ctype[i] =='S' ):
                        S_T += 1
                        continue
                    
                    if(Bond_data_1statom_Ctype[i] == 'S' and Bond_data_2ndatom_Ctype[i] =='Q' ):
                        S_Q += 1
                        continue
                    
                    if(Bond_data_1statom_Ctype[i] == 'Q' and Bond_data_2ndatom_Ctype[i] =='S' ):
                        S_Q += 1
                        continue
                    
                    if(Bond_data_1statom_Ctype[i] == 'T' and Bond_data_2ndatom_Ctype[i] =='T' ):
                        T_T += 1
                        continue
                    
                    if(Bond_data_1statom_Ctype[i] == 'T' and Bond_data_2ndatom_Ctype[i] =='Q' ):
                        T_Q += 1
                        continue
                    
                    if(Bond_data_1statom_Ctype[i] == 'Q' and Bond_data_2ndatom_Ctype[i] =='T' ):
                        T_Q += 1
                        continue
                    
                    if(Bond_data_1statom_Ctype[i] == 'Q' and Bond_data_2ndatom_Ctype[i] =='Q' ):
                        Q_Q += 1
                        continue

                #############Processing for the P,S,T,Q-H boBond_detail_dataframend########################

                #Combined list of First and second col
                Bond_data_0n1_col_list = list(Bond_data_1statoms) + list(Bond_data_2ndatoms)

                #Counter of repeatation of atom in list in terms of dictionary
                Bond_data_0n1_counter = collections.Counter(Bond_data_0n1_col_list)

                ### Actual Ctype-H Calcualtion
                for i in range(Smiles_length): 
                    Bond_formed = Bond_data_0n1_counter.get(i+1) #+1 as i starts from zero
                    if (Smiles_char[i] == 'C'):
                        remaining_bond = 4 - Bond_formed
                        if (Smile_dict.get(i+1) == 'P'):
                            P_H = P_H + remaining_bond
                            continue
                        if (Smile_dict.get(i+1) == 'S'):
                            S_H = S_H + remaining_bond
                            continue
                        if (Smile_dict.get(i+1) == 'T'):
                            T_H = T_H + remaining_bond
                            continue
                        if (Smile_dict.get(i+1) == 'Q'):
                            Q_H = Q_H + remaining_bond
                            # exit(1)
                            continue
                    else:
                        continue
                
                #Combining data into the list to append in the dataset 
                #Here I have total 13 feature and to make sequence let's run loop 
                #Order : 'P_P','P_S','P_T','P_Q','S_S','S_T','S_Q','T_T','T_Q','Q_Q','P_H','S_H','T_H'
                carbon_type_n_bond_list = [P_P,P_S,P_T,P_Q,S_S,S_T,S_Q,T_T,T_Q,Q_Q,P_H,S_H,T_H]
                Final_list_type_n_bond = carbon_type_array + carbon_type_n_bond_list    #Adding list oc carbon type and bond
                Final_list_type_n_bond = pd.Series(Final_list_type_n_bond,index=columns) #Conversion into series to add in dataframe
                Bond_detail_dataframe=Bond_detail_dataframe.append(Final_list_type_n_bond,ignore_index=True)

        #Adding Fuel name to dataset 
        Unique_fuel_smiles = pd.Series(Unique_fuel_smiles)
        Bond_detail_dataframe['Fuel'] = Unique_fuel_smiles

        return Bond_detail_dataframe