import pandas as pd
import numpy as np

def count_mutations(allele_data, mutations, sample_resitance_mutations = None, sample_list = None, \
                    gene = 'k13', split_microhaps = True):
    """
    This method counts the numer of mutations (of a single locus) found for each sample. 
    
    Parameters: 
    -----------
    allele_data: pd.DataFrame
        Data frame with information of sample, gene and microhaplotypes. 
    mutations: pd.DataFrame
        Data frame describing the mutations (Name, gene, and microhaplotype index, reference and resistance mutation). 
    sample_resistance_mutations: pd.DataFrame
        Data containing the appearences of each mutation per sample. 
    sample_list: list
        List of all samples to be used. 
    gene: str
        Name of the gene to analyse. 
    split_microhaps: bool
        If True, the microhaplotypes are split to treat all loci individually. 
    
    Returns: 
    --------
    sample_resistance_mutations: pd.DataFrame
        Data containing the appearences of each mutation per sample. 
    """
    if sample_list is None: 
        sample_list = allele_data['SampleID'].unique()
    if sample_resitance_mutations is None: 
        mutation_names = [gene + '_' + i for i in mutations.loc['Name'].unique()]
        sample_resitance_mutations = pd.DataFrame(index = sample_list, columns = mutation_names)
    gene_mask = allele_data['Gene'] == gene
    for sample in sample_list:
        #select sample
        sample_mask = allele_data['SampleID'] == sample
        #select gene     
        if np.sum(sample_mask&gene_mask) > 0:
            #for all entries of this sample and gene
            for i in range(len(allele_data[sample_mask&gene_mask])):
                microhap_data = allele_data[sample_mask&gene_mask].iloc[i]
                if split_microhaps:
                    sample_resitance_mutations = microhap2mutation_per_locus(microhap_data, mutations, sample_resitance_mutations)
                else:
                    sample_resitance_mutations = microhap2mutation_data(microhap_data, mutations, sample_resitance_mutations)#TODO do this
    return sample_resitance_mutations

def microhap2mutation_data(microhap_data, mutations, sample_resitance_mutations):
    """
    Thi method counts the mutations appeared in a given microhaplotype. 
    
    Parameters: 
    -----------
    microhap_data: pd.DataFrame
        Data including information on the index, reference and detected marker.
    mutations: pd.DataFrame
        Data frame describing the mutations (Name, gene, and microhaplotype index, reference and resistance mutation). 
    sample_resistance_mutations: pd.DataFrame
        Data containing the appearences of each mutation per sample. 
    
    Returns: 
    --------
    sample_resistance_mutations: pd.DataFrame
        Data containing the appearences of each mutation per sample.
    """
    sample = microhap_data['SampleID']
    gene = microhap_data['Gene']
    #microhap_data['MicrohapIndex']
    #microhap_data['RefMicrohap']
    #microhap_data['Microhaplotype']
    for mut in mutations.columns:
        if mutations[mut]['MicrohapIndex'] == microhap_data['MicrohapIndex']:
            #If it corresponds to the reference
            if mutations[mut]['RefMicrohap'] == microhap_data['Microhaplotype']:
                #If NA, assign 0
                if np.isnan(sample_resitance_mutations.loc[sample, gene + '_' + mut]):
                    sample_resitance_mutations.loc[sample, gene + '_' + mut] = 0
            #If it corresponds to the resistance
            elif mutations[mut]['ResMicrohap'] == microhap_data['Microhaplotype']:
                #If NA, assign 1, else +=1
                if np.isnan(sample_resitance_mutations.loc[sample, gene + '_' + mut]):
                    sample_resitance_mutations.loc[sample, gene + '_' + mut] = 1
                else:
                    sample_resitance_mutations.loc[sample, gene + '_' + mut] += 1
            #If neither reference or resistance
            else:
                #If NA, assign 0
                if np.isnan(sample_resitance_mutations.loc[sample, gene + '_' + mut]):
                    sample_resitance_mutations.loc[sample, gene + '_' + mut] = 0
    return sample_resitance_mutations

def microhap2mutation_per_locus(microhap_data, mutations, sample_resitance_mutations):
    """
    Thi method counts the mutations appeared in a given microhaplotype at each locus. 
    
    Parameters: 
    -----------
    microhap_data: pd.DataFrame
        Data including information on the index, reference and detected marker.
    mutations: pd.DataFrame
        Data frame describing the mutations (Name, gene, and microhaplotype index, reference and resistance mutation). 
    sample_resistance_mutations: pd.DataFrame
        Data containing the appearences of each mutation per sample. 
    
    Returns: 
    --------
    sample_resistance_mutations: pd.DataFrame
        Data containing the appearences of each mutation per sample.
    """
    sample = microhap_data['SampleID']
    gene = microhap_data['Gene']
    #Split combined markers
    microhap_index_list = microhap_data['MicrohapIndex'].split('/')
    ref_microhap_list = microhap_data['RefMicrohap'].split('/')
    microhap_list  = microhap_data['Microhaplotype'].split('/')
    #check mutations and assign them for each marker
    for i,m in enumerate(microhap_index_list):
        #check in what mutations the marker locus coincides
        for mut in mutations.columns:
            if m == mut[1:-1]:
                #If it corresponds to the reference
                if ref_microhap_list[i] == microhap_list[i]:
                    #If NA, assign 0
                    if np.isnan(sample_resitance_mutations.loc[sample, gene + '_' + mut]):
                        sample_resitance_mutations.loc[sample, gene + '_' + mut] = 0
                #If it corresponds to the resistance
                elif microhap_list[i] == mutations[mut]['ResMicrohap']:
                    #If NA, assign 1, else +=1
                    if np.isnan(sample_resitance_mutations.loc[sample, gene + '_' + mut]):
                        sample_resitance_mutations.loc[sample, gene + '_' + mut] = 1
                    else:
                        sample_resitance_mutations.loc[sample, gene + '_' + mut] += 1
                #If neither reference or resistance
                else:
                    #If NA, assign 0
                    if np.isnan(sample_resitance_mutations.loc[sample, gene + '_' + mut]):
                        sample_resitance_mutations.loc[sample, gene + '_' + mut] = 0
    return sample_resitance_mutations


#K13 data
k13_validated_markers_list = ['F446I', 'N458Y', 'M476I', 'Y493H', 'R539T', 'I543T', 'P553L', \
                              'R561H', 'P574L', 'C580Y']
#notice that C469 and N537 apperar twice for having 2 options
k13_candidate_markers_list = ['P441L', 'G449A', 'C469F', 'C469Y', 'A481V', 'R515K', 'P527H', \
                              'N537I', 'N537D', 'G538V', 'V568G', 'R622I', 'A675V']
k13_mutations_list = k13_validated_markers_list + k13_candidate_markers_list
k13_mutations = pd.DataFrame({
})
for mutation in k13_mutations_list:
    k13_mutations[mutation] = {'Name' : mutation,
    'Gene' : 'k13', 
    'MicrohapIndex' : mutation[1:-1], 
    'RefMicrohap' : mutation[0], 
    'ResMicrohap' : mutation[-1]}

# pfdhfr codons
dhfr_mutations_list = ['A16V', 'C50R', 'N51I', 'C59R', 'S108N', 'S108T']
dhfr_mutations = pd.DataFrame({
})
for mutation in dhfr_mutations_list:
    dhfr_mutations[mutation] = {'Name' : mutation,
    'Gene' : 'dhfr', 
    'MicrohapIndex' : mutation[1:-1], 
    'RefMicrohap' : mutation[0], 
    'ResMicrohap' : mutation[-1]}
    
# pfdhps codons
dhps_mutations_list = ['I431V', 'S436A', 'S436F', 'A437G', 'K540E', 'K540N', 'A581G', 'A613S', 'A613T']
dhps_mutations = pd.DataFrame({
})
for mutation in dhps_mutations_list:
    dhps_mutations[mutation] = {'Name' : mutation,
    'Gene' : 'dhps', 
    'MicrohapIndex' : mutation[1:-1], 
    'RefMicrohap' : mutation[0], 
    'ResMicrohap' : mutation[-1]}

# pfmdr1
mdr1_mutations_list = ['N86Y', 'Y184F', 'S1034C', 'N1042D', 'D1246Y']
mdr1_mutations = pd.DataFrame({
})
for mutation in mdr1_mutations_list:
    mdr1_mutations[mutation] = {'Name' : mutation,
    'Gene' : 'mdr1', 
    'MicrohapIndex' : mutation[1:-1], 
    'RefMicrohap' : mutation[0], 
    'ResMicrohap' : mutation[-1]}

# pfcrt 
crt_mutations = pd.DataFrame({
})
crt_mutations['CVMNK72-76CVIET'] = {
    'Name' : 'CVMNK72-76CVIET', 
    'Gene' : 'crt', 
    'MicrohapIndex' : '72/73/74/75/76', 
    'RefMicrohap' : 'C/V/M/N/K', 
    'ResMicrohap' : 'C/V/I/E/T'}