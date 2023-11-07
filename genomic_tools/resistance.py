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

def get_quintuple_sextuple_mutations(allele_data, all_mut_counts):
    """
    This method obtains the dhfr-dhps quintuple and sextuple mutations. 
    
    Parameters: 
    -----------
    allele_data: pd.DataFrame
        Data frame with information of sample, gene and microhaplotypes.
    all_mut_counts: pd.DataFrame
        Data containing the appearences of each mutation per sample. 
    
    Returns: 
    --------
    all_mut_counts: pd.DataFrame
        Data containing the appearences of each mutation per sample, 
        including the quintuple and sextuple mutations. 
    """
    # Allele frequencies of dhfr-dhps quintuple/sextuple mutant loci
    dhfr_quint_mut_loci = ['108/164', '16/51/59']
    dhps_quint_mut_loci = ['431/436/437', '540/581']
    for sample in all_mut_counts.index:
        dhfr_mask = allele_data['Gene'] == 'dhfr'
        dhps_mask = allele_data['Gene'] == 'dhps'
        sample_mask = allele_data['SampleID'] == sample
        for locus in dhfr_quint_mut_loci:
            mask_locus = allele_data['MicrohapIndex'] == locus
            all_mut_counts.loc[sample, 'dhfr_' + locus + '_nalleles'] = allele_data[dhfr_mask&sample_mask&mask_locus]['n.alleles'].mean()
        for locus in dhps_quint_mut_loci:
            mask_locus = allele_data['MicrohapIndex'] == locus
            all_mut_counts.loc[sample, 'dhps_' + locus + '_nalleles'] = allele_data[dhps_mask&sample_mask&mask_locus]['n.alleles'].mean()
    
    #Boolean confirming if mutation is present regarless of allele numbers in the loci (we need at most one microhap with >1 alleles)
    allele_confirmation = (all_mut_counts[['dhfr_108/164_nalleles', 'dhfr_16/51/59_nalleles', \
                                           'dhps_431/436/437_nalleles', 'dhps_540/581_nalleles']] > 1).sum(axis = 1) < 2
    allele_confirmation[allele_confirmation == 0] = np.nan
    
    #Mutation present
    all_mut_counts['dhfr-dhps quint'] = all_mut_counts['dhfr_S108N']*all_mut_counts['dhfr_N51I']* all_mut_counts['dhfr_C59R']*all_mut_counts['dhps_A437G']*all_mut_counts['dhps_K540E']*allele_confirmation
    all_mut_counts['dhfr-dhps sext'] = all_mut_counts['dhfr-dhps quint']*all_mut_counts['dhps_A581G']*allele_confirmation
    all_mut_counts['dhfr-dhps quint'][all_mut_counts['dhfr-dhps quint'] > 1] = 1
    all_mut_counts['dhfr-dhps sext'][all_mut_counts['dhfr-dhps sext'] > 1] = 1
    
    return all_mut_counts

def count_all_mutations(allele_data, return_all = False):
    """
    This method calculates all the presence of drug resistance mutations in an allele data table. 
    
    Parameters: 
    -----------
    allele_data: pd.DataFrame
        Data frame with information of sample, gene and microhaplotypes.
    return_all: bool
        If True, in addition to a data frame with all mutations if returns a separate data frame 
        for each type of mutation. 
        
    Returns:
    --------
    all_mut_counts: pd.DataFrame
        Data containing the appearences of each mutation per sample. 
    k13_mut_counts: pd.DataFrame
        Data containing the appearences of each Kelch 13 mutation.
    dhfr_mut_counts: pd.DataFrame
        Data containing the appearences of each dhfr mutation.
    dhps_mut_counts: pd.DataFrame
        Data containing the appearences of each dhps mutation.
    mdr1_mut_counts: pd.DataFrame
        Data containing the appearences of each mdr1 mutation.
    crt_mut_counts: pd.DataFrame
        Data containing the appearences of crt mutations.
    """
    k13_mut_counts = count_mutations(allele_data, k13_mutations, sample_list = None, gene = 'k13')
    k13_mut_counts['total_k13'] = k13_mut_counts.sum(axis = 1, min_count = 1)
    k13_mut_counts['k13'] = 1*k13_mut_counts['total_k13']
    k13_mut_counts['k13'][k13_mut_counts['k13']>1] = 1
    dhfr_mut_counts = count_mutations(allele_data, dhfr_mutations, sample_list = None, gene = 'dhfr')
    dhfr_mut_counts['total_dhfr'] = dhfr_mut_counts.sum(axis = 1, min_count = 1)
    dhfr_mut_counts['dhfr'] = 1*dhfr_mut_counts['total_dhfr']
    dhfr_mut_counts['dhfr'][dhfr_mut_counts['dhfr']>1] = 1
    dhps_mut_counts = count_mutations(allele_data, dhps_mutations, sample_list = None, gene = 'dhps')
    dhps_mut_counts['total_dhps'] = dhps_mut_counts.sum(axis = 1, min_count = 1)
    dhps_mut_counts['dhps'] = 1*dhps_mut_counts['total_dhps']
    dhps_mut_counts['dhps'][dhps_mut_counts['dhps']>1] = 1
    mdr1_mut_counts = count_mutations(allele_data, mdr1_mutations, sample_list = None, gene = 'mdr1')
    mdr1_mut_counts['total_mdr1'] = mdr1_mut_counts.sum(axis = 1, min_count = 1)
    mdr1_mut_counts['mdr1'] = 1*mdr1_mut_counts['total_mdr1']
    mdr1_mut_counts['mdr1'][mdr1_mut_counts['mdr1']>1] = 1
    crt_mut_counts = count_mutations(allele_data, crt_mutations, sample_list = None, gene = 'crt', split_microhaps=False)
    
    #Merging all
    all_mut_counts = pd.merge(k13_mut_counts, dhfr_mut_counts, left_index = True, right_index = True, how = 'outer')
    all_mut_counts = pd.merge(all_mut_counts, dhps_mut_counts, left_index = True, right_index = True, how = 'outer')
    all_mut_counts = pd.merge(all_mut_counts, mdr1_mut_counts, left_index = True, right_index = True, how = 'outer')
    all_mut_counts = pd.merge(all_mut_counts, crt_mut_counts, left_index = True, right_index = True, how = 'outer')
    
    #obtain phfr-dhps quintuple and sextuple mutations
    all_mut_counts = get_quintuple_sextuple_mutations(allele_data, all_mut_counts)
    
    if return_all:
        return all_mut_counts, k13_mut_counts, dhfr_mut_counts, dhps_mut_counts, mdr1_mut_counts, crt_mut_counts
    else:
        return all_mut_counts