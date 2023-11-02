import pandas as pd
import numpy as np

def count_mutations(allele_data, mutations, sample_resitance_mutations = None, sample_list = None, gene = 'k13'):
    """
    DOCUMENT
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
                sample_resitance_mutations = microhap2mutation_data(microhap_data, mutations, sample_resitance_mutations)
    return sample_resitance_mutations

def microhap2mutation_data(microhap_data, mutations, sample_resitance_mutations):
    """
    DOCUMENT
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
