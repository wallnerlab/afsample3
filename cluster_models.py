import glob
import shutil, os, sys, re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import gemmi
from scipy import stats
from scipy.spatial import distance

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import minmax_scale

import MDAnalysis as mda
from MDAnalysis.analysis.dssp import DSSP

import pymol
from parallelbar import progress_map
from functools import partial
import pickle
import kmedoids
import numpy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler


def k_medoids(X, l, labels, k=3, max_iter=100):
    """
    K-Medoid algorithm to find suitable representative structures from each cluster defined by HDBSCAN.

    Input:
        X:        np.ndarray (n, m)  | all points from one HDBSCAN cluster
        k:        number of medoids  | 
        max_iter: maximum number of iterations allowed to minimize the distance
        l:        current HDBSAN label
        labels:   full list of HDBSCAN labels

    Output:
        medoids:     indices of the K medoids
        total_cost:  sum of distances of each point to its medoid
    """
    np.random.seed(42)

    #start with random k points
    temp = X.copy()
    mask = np.zeros(X.shape, dtype=bool)
    mask[np.argwhere(labels == l)] = True
    label_idx=np.argwhere(labels == l).flatten()
    #check the number of points in a cluster
    #if less than 4 just return those indices
    _, cluster_count = np.unique(mask[:,0], return_counts=True) # count = False, True
    cluster_count = cluster_count[[idx for idx, val in enumerate(_) if val == True][0]]#<-account for the case of one cluster

    if cluster_count < 4:
        print(f"only {cluster_count} points in cluster {l}, returning all points")
        return np.ravel(np.argwhere(mask[:,0] == True)), np.nan
    # block out values that are not within the current HDBSCAN group
    temp[~mask] = 0 #99999

    number_samples = temp.shape[0]
    #medoids = np.random.choice(number_samples, k, replace=False)
    medoids = np.random.choice(label_idx, k, replace=False)
    #print(medoids)
    #distance matrix of randomly chosen points
    D = distance.cdist(temp, temp[medoids], metric='euclidean')
    tot_cost = np.sum(np.min(D[label_idx], axis=1))

    itr = 0
    while itr < max_iter:
        reduced = False

        #loop through all possibilities
        for m_idx in range(k):
            for current_idx in label_idx: #range(number_samples):
                if current_idx in medoids:
                    continue

                new_medoids = medoids.copy()
                new_medoids[m_idx] = current_idx

                #new distance matrix
                D_new = distance.cdist(temp, temp[new_medoids], metric='euclidean')
                new_cost = np.sum(np.min(D_new[label_idx], axis=1))

                #if the cost has been reduced move onto the the next sample
                if new_cost < tot_cost:
                    medoids = new_medoids
                    tot_cost = new_cost
                    reduced = True
                    break
            if reduced:
                break

        if not reduced:
            #If there was no improvement we should be converged
            break
        itr+=1
    return medoids, tot_cost

def k_medoids_(X, weights=None, k=3, max_iter=100):
    """
    K-Medoid algorithm to find suitable representative structures from each cluster defined by HDBSCAN.

    Input:
        X:        np.ndarray (n, m)  | all points from one HDBSCAN cluster
        k:        number of medoids  | 
        max_iter: maximum number of iterations allowed to minimize the distance
        l:        current HDBSAN label
        labels:   full list of HDBSCAN labels

    Output:
        medoids:     indices of the K medoids
        total_cost:  sum of distances of each point to its medoid
    """
    np.random.seed(42)
    
    #start with random k points
    temp = X.copy()
    number_samples = temp.shape[0]
    label_idx=np.arange(number_samples)
    #check the number of points in a cluste
    #if less than 4 just return those indices
    #_, cluster_count = np.unique(mask[:,0], return_counts=True) # count = False, True
    #cluster_count = cluster_count[[idx for idx, val in enumerate(_) if val == True][0]]#<-account for the case of one cluster

    #if cluster_count < 4:
    #    print(f"only {cluster_count} points in cluster {l}, returning all points")
    #    return np.ravel(np.argwhere(mask[:,0] == True)), np.nan
    # block out values that are not within the current HDBSCAN group
    #temp[~mask] = 0 #99999
    if number_samples < k:
        print(f"only {number_of_samples} points in data returning all points")
        return label_idx


  
    medoids = np.random.choice(number_samples, k, replace=False)
 
    D = distance.cdist(temp, temp[medoids], metric='euclidean')
    tot_cost = np.sum(np.min(D, axis=1))

    itr = 0
    while itr < max_iter:
        reduced = False

        #loop through all possibilities
        for m_idx in range(k):
            for current_idx in range(number_samples):
                if current_idx in medoids:
                    continue

                new_medoids = medoids.copy()
                new_medoids[m_idx] = current_idx

                #new distance matrix
                D_new = distance.cdist(temp, temp[new_medoids], metric='euclidean')
                new_cost = np.sum(np.min(D_new, axis=1))
                
                #if the cost has been reduced move onto the the next sample
                if new_cost < tot_cost:
                    medoids = new_medoids
                    tot_cost = new_cost
                    print(medoids,tot_cost)
                    reduced = True
                    break
            if reduced:
                break

        if not reduced:
            #If there was no improvement we should be converged
            break
        itr+=1
    return medoids, tot_cost


import tempfile

def run_foldseek(file,db_directory='UNDEF',ext='.pdb',outpath='UNDEF'):
    outfile=file.replace(ext,"-self.foldseek")

    with tempfile.TemporaryDirectory() as tmpdir:
        foldseek_run = ["foldseek", "easy-search", file, db_directory + "DB", outfile, tmpdir, "--threads", "2","--format-mode", "0", "--format-output", "query,target,alntmscore,qaln,taln,alnlen,evalue,ttmscore", "--exhaustive-search", "1", "-s", "9.5"]
        if not os.path.isfile(outfile):
           # print(f"running foldseek on {' '.join(foldseek_run)}")
            try:
                response = subprocess.run(foldseek_run, capture_output=True, text=True, check=True)
               # print(response.check_returncode())
            except subprocess.CalledProcessError as e:
                print("foldseek failed to run {:}".format(file))
                print("Error:", e.stderr)
                return None
           # print('{:} succeeded!!!'.format(file))
        #else:
        #    print("{:} already exists".format(outfile))
        return outfile

def run_all_foldseek(pdb_files, outpath, n_cpu=4, references=False): #outpath):
# def main():
    """
    requires Foldseek and Pymol

    Find all pdb files from CF-Random generated directories.
    This script will automatically generate a Foldseek database of these structures 
    then calculate a similarity matrix of all structures based on bit-score.
    similarity matrix -> PCA -> HDBSCAN -> K-medoids -> structures of interest.

    The final output is then a png file showing the result of PCA and HDBSCAN
    a text file containing the coordinates of the structures of interest, file name, and group ID
    finally this script will automatically generate a pse file of the structures_of_interest
    """

    results_data={}
    foldseek_ids={}
    #_______________collect all pdb files that CF-Random generated_____________________________
    db_directory = outpath +  "/pdbs_for_db/"
    #db_directory =  "/pdbs_for_db/"
    if not os.path.isdir(db_directory):
        os.mkdir(db_directory)
    #pdb_files = glob.glob("./**/*.pdb", recursive=True)
    #pdb_files = glob.glob(blind_path + "/**/*.pdb", recursive=True)
    #pdb_files = [file for file in pdb_files if db_directory not in file]
    print("Gathering pdb files for self-search")
    foldseek_pdbs = {}
    for file in pdb_files:
        dest_name = file.replace('/','-')
        absfile=os.path.abspath(file)
        #print(file,dest_name)
        if dest_name[0]=='-':
            dest_name=dest_name[1:] #remove first -
        foldseek_ids[file]=os.path.splitext(dest_name)[0]
    
        new_pdb_file= db_directory + dest_name

        if not os.path.isfile(new_pdb_file):
            #shutil.copyfile(file, new_pdb_file)
            
            os.symlink(absfile,new_pdb_file)
          
        #new_pdb_files.append(new_pdb_file)
        foldseek_pdbs[file]=new_pdb_file
    #__________________________________________________________________________________________




    if not references:
        print("Creating database...")
        create_db = ["foldseek", "createdb", db_directory, db_directory + "DB"]
        if not os.path.isfile(db_directory + "DB"):
            try:
                response = subprocess.run(create_db, capture_output=True, text=True, check=True )
            except subprocess.CalledProcessError as e:
                print("ERROR:\n", e.stderr)
            
            print('Success database is up!')
        else:
            print("found an existing DB")

    ext='.'+pdb_files[0].split('.')[-1]
    run_foldseek_db = partial(run_foldseek,db_directory=db_directory,ext=ext,outpath=outpath)
        

    #________________Calculate foldseek self comparison of all predicted structures____________
    print("Foldseek all-against-all")
    foldseek_out=progress_map(run_foldseek_db, foldseek_pdbs.values(),n_cpu=n_cpu)
    #print(f"extension is {ext}")
    #_________________________________________
    #pdb2fol
 
    foldseek_outputs=dict(zip(foldseek_pdbs.keys(),foldseek_out))
  

    results_data={'foldseek_ids':foldseek_ids,
                  'foldseek_pdbs':foldseek_pdbs,
                  'foldseek_outputs':foldseek_outputs}
    
    return results_data

import warnings

# Suppress UserWarning coming specifically from MDAnalysis.coordinates.PDB module
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"MDAnalysis\."
)

def count_secondary_structure(pdbfile):
      u = mda.Universe(pdbfile)
      s = DSSP(u).run().results.dssp[0]
      return [np.char.count(s,char).sum().astype(int) for char in ['-','E','H']]




def get_loop_outliers(pdbfiles): #,ext='.pdb'): #outpath):
    #__________Populate a correlation matrix with bit scores_______________________________________________
#if 1==1:
    #everything will be sorted by the text of the file name
    print('Assigning secondary structure counts to all pdb files and removing outliers')
    #files = glob.glob(outpath + "/**/*-self.foldseek")
    #pdbfiles = [file.replace("-self.foldseek",ext) for file in files]
    ss_count = progress_map(count_secondary_structure, pdbfiles,n_cpu=32)
    #first remove any outliers from the dssp loop distribution, they tend to be unfolded predictions
    #files_dssp = [];files_count = [];
   # ss_count = []
   # for i,file in enumerate(files):
   #     print(i,end=", ",flush=True)
   #     pdbfile=file.replace("-self.foldseek",ext)
   #     u = mda.Universe(pdbfile)
   #     s = DSSP(u).run().results.dssp[0]

    #    ss_count.append([np.char.count(s,char).sum().astype(int) for char in ['-','E','H']])

    
      #  dssp, count = np.unique(s, return_counts=True)
        # ['-' 'E' 'H']
      #  if len(dssp) < 3:
      #      if '-' not in dssp:
      #          dssp  = np.insert(dssp, 0 ,'-')
      #          count = np.insert(count,0, 0)
      #      if 'E' not in dssp:
      #          dssp  = np.insert(dssp, 1 ,'E')
      #          count = np.insert(count,1, 0)
      #      if 'H' not in dssp:
      #          dssp  = np.insert(dssp, 2 ,'H')
      #          count = np.insert(count,2, 0)
      #  files_dssp.append(dssp)
      #  files_count.append(count)
    ss_count = np.array(ss_count)    
    z_scores = stats.zscore(ss_count[:, 0]) #JUST USE THE LOOP COUNT
    #files_dssp = np.array(files_dssp);
    #files_count = np.array(files_count);
    #z_scores = stats.zscore(files_count[:, 0])
    outlier_idx = np.argwhere(z_scores > 3).flatten()
    #print(outlier_idx)
    outliers=[pdbfiles[i] for i in outlier_idx]
    return(outliers)
    # remove unfolded proteins from file list 

  #  files = np.array(files)
  #  mask = np.zeros(files.shape, dtype=bool)
  #  mask[outlier_idx] = True
  #  for file in files[mask]:
  ##      print("removed from analysis: ",file.replace("-self.foldseek",ext))
   # files = files[~mask]
   # files = sorted(files)
   # files_pdb = [file.replace("-self.foldseek",ext) for file in files]
    #files_pdb = [file.replace("-self.foldseek",ext) for file in files]
   # return files, files_pdb
    
#orm_corr_mtx=populate_corr_mtx(results_data, outliers=outliers)
def populate_corr_mtx(results_data, outliers=None, ext='.pdb', references=None, foldseek_keys_n=None): 
    #files,files_pdb=remove_outliers(outpath,ext=ext)
     
    #foldseek_keys=[os.path.basename(f).replace(ext,'') for f in files_pdb]

    if not foldseek_keys_n:
        foldseek_keys=[(results_data['foldseek_ids'][i],results_data['foldseek_outputs'][i],i) #id,output,pdb
                    for i in results_data['foldseek_ids'].keys() 
                    if i not in outliers] 
    
        N=len(foldseek_keys)
        corr_mtx= np.zeros((N,N),dtype=int)
        foldseek_keys_index = {key: i for i, (key,_,_) in enumerate(foldseek_keys)} 
        
    else:
        foldseek_keys=[(results_data['foldseek_ids'][i],results_data['foldseek_outputs'][i],i) #id,output,pdb
                    for i in results_data['foldseek_ids'].keys() 
                    if i not in outliers] 
        N=len(foldseek_keys)
        corr_mtx= np.zeros((N, len(foldseek_keys_n)),dtype=int)
        
        foldseek_keys_index_n = {key: i for i, (key,_,_) in enumerate(foldseek_keys_n)} 
        foldseek_keys_index = {key: i for i, (key,_,_) in enumerate(foldseek_keys)}  

    if references:
        print('foldseek_keys', len(foldseek_keys))
        print('foldseek_keys_index', len(foldseek_keys_index_n))

    for _,foldseek_output,_ in foldseek_keys:
        #print(i,end=", ",flush=True)
        with open(foldseek_output, 'r') as f:
            for line in f:
                cols = line.rstrip().split()
                query=cols[0]
                hit=cols[1]
                #bitscore=int(cols[-1])   
                bitscore=int(float(cols[-1])*100)

                if references:
                    searchspace = foldseek_keys_index_n
                else:
                    searchspace = foldseek_keys_index

                if hit in searchspace: #query is filter in foldseek_keys
                    if references: 
                        i=foldseek_keys_index[query]
                        j=searchspace[hit]
                        # print(query, hit, line, bitscore)
                        # sys.exit()
                    else:
                        i=searchspace[query]
                        j=searchspace[hit]

                    #print(i, j, corr_mtx.shape)

                    corr_mtx[i,j] = bitscore
                    if corr_mtx[i,j] == -2147483648:   #bug in foldseek occasionally returns -2,147,483,648
                        corr_mtx[i,j] = 0   
    
    return corr_mtx, foldseek_keys
    # norm_corr_mtx = minmax_scale(corr_mtx, axis=1)
    # norm_corr_mtx = (norm_corr_mtx + norm_corr_mtx.T) /2
    
    # return {'pdbfiles':[i[2] for i in foldseek_keys],
    #         'mtx':norm_corr_mtx}

def scale_norm(corr_mtx, foldseek_keys, scaler=None):
    if not scaler:
        scaler = MinMaxScaler()
        norm_corr_mtx = scaler.fit_transform(corr_mtx)
        return {'pdbfiles':[i[2] for i in foldseek_keys],
            'mtx':norm_corr_mtx}, scaler
    print('corr matrix shape', corr_mtx.shape)
    norm_corr_mtx = scaler.transform(corr_mtx)    
    return {'pdbfiles':[i[2] for i in foldseek_keys],
            'mtx':norm_corr_mtx}, scaler

import pandas as pd

class Clustering:
    def __init__(self, outpath, name, k, n_components, norm_corr_mtx, norm_corr_mtx_ref=None):
        self.outpath = outpath
        self.name = name
        self.k = k
        self.n_components = n_components
        self.path_to_tmdf = '/proj/wallner-b/users/x_yogka/AFsample3/af3-dev/src/analysis/notebooks/intermediates/analysis_csvs_v3'

        # Ensemble
        self.norm_corr_mtx = norm_corr_mtx['mtx']
        self.pdbfiles = norm_corr_mtx['pdbfiles']

        # Refernce
        if norm_corr_mtx_ref:
            self.norm_corr_mtx_ref = norm_corr_mtx_ref['mtx']
            self.refs = norm_corr_mtx_ref['pdbfiles']

    def run_pca(self, plot=True, show_plot=False):
        sklearn_pca = PCA(n_components=self.n_components)
        pca = sklearn_pca.fit_transform(self.norm_corr_mtx)
        return sklearn_pca, pca

    def main(self, map_references=False):
        self.read_tmout_with_intermediates()
        sklearn_pca, pca = self.run_pca()
        labels,_ = self.cluster_structures(pca)
        labels_df = pd.DataFrame([self.pdbfiles, labels])
        print(np.unique(labels))
        if map_references:
            pcaref = sklearn_pca.transform(self.norm_corr_mtx_ref)
            print('pca ref:', self.norm_corr_mtx_ref.shape, pcaref.shape)
            self.annotate_points_in_pca(pca, pcaref, labels)
        else:
            self.annotate_points_in_pca(pca=pca, pcaref=[], labels=labels)
    
    def annotate_points_in_pca(self, pca, pcaref, labels):
        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        c = ax[0].scatter(pca[:,0], pca[:,1], c=labels, cmap='viridis', s=20)
        plt.colorbar(c, label='Cluster Label')
        if not len(pcaref)==0:
            ax[0].scatter(pcaref[:,0], pcaref[:,1],marker='D', s=30)    #plot ref pdbs

        files_of_interest, pca_of_interest = [], []
        for l in np.unique(labels):
            label_idx=np.argwhere(labels == l).flatten()
            fp = kmedoids.fastpam1(euclidean_distances(pca[label_idx]), self.k, 100,random_state=42)
            kmed_idx=label_idx[fp.medoids]
            for idx in kmed_idx:
                files_of_interest.append([self.pdbfiles[idx], l, labels[idx]])
                pca_of_interest.append(pca[idx])
                ax[0].plot(pca[idx,0], pca[idx,1], 'r*', markersize=15) #highlight the medoids on the PCA plot

        prefix=f'{self.outpath}/{self.name}-k{self.k}-pca_n{self.n_components}'
        ax[0].set_xlabel('PCA 1')
        ax[0].set_ylabel('PCA 2')
        ax[0].set_title(f'PCA of structures colored by HDBSCAN clusters with K-medoids stars (k={self.k})')
        plt.savefig(f'{prefix}-cluster.png')
        print(f'Saved figure to: {prefix}-cluster.png')
    
    def cluster_structures(self, X):
        """
        loop through values of k and define best value of k with silhouette_score

        Input: 
            X : np.ndarray (n, m) | result of PCA

        Output: 
            cluster_labels : (n, 1) | list of optimal clusters for X
        """
        k_range = range(2,51)
        sil_score = []
        for k in k_range:
            clustering = HDBSCAN(min_cluster_size=k,min_samples=1)
            clustering.fit(X)
            if len(set(clustering.labels_)) > 1 and len(set(clustering.labels_)) < len(X):
                score = silhouette_score(X, clustering.labels_, metric='euclidean')
                sil_score.append(score)
            else:
                sil_score.append(-1)

        opt_k = k_range[np.argmax(sil_score)]
        clustering = HDBSCAN(min_cluster_size=opt_k)
        clustering.fit(X)
        return clustering.labels_,clustering

    def read_tmout_with_intermediates(self):
        df = pd.read_csv(f'{self.path_to_tmdf}/intermediates_{self.name}.csv')
        print(df.shape, df.columns)
        print(df.model[:2].values)
        renamed_pdbfiles = [a.replace('allmodels/', '').replace('_model', '/model') for a in self.pdbfiles[:2]]
        print(self.pdbfiles[:2])

#def pca_and_cluster(outpath,name,files=None,norm_corr_mtx=None,ext='.pdb',n_components=4,k=3,show_plot=False):
def pca_and_cluster(outpath,name,norm_corr_mtx,n_components=4,k=3,show_plot=False):
   # if norm_corr_mtx is None or files is None:
   #     files,files_pdb,norm_corr_mtx=populate_corr_mtx(outpath, ext=ext)
    pdbfiles=norm_corr_mtx['pdbfiles']
    sklearn_pca = PCA(n_components=n_components)
    pca = sklearn_pca.fit_transform(norm_corr_mtx['mtx'])
    labels,_= cluster_structures(pca)
    
    plt.figure(figsize=(8,6))
    plt.scatter(pca[:,0], pca[:,1], c=labels, cmap='viridis', s=20)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title(f'PCA of structures colored by HDBSCAN clusters with K-medoids stars (k={k})')
    plt.colorbar(label='Cluster Label')

     #find the structures_of_interest
    files_of_interest = []
    pca_of_interest = []
    for l in np.unique(labels):
        #print(k)
        #pca_labels=
        label_idx=np.argwhere(labels == l).flatten()
        #kmed_idx, tot_cost  = k_medoids(pca, l, labels,k=k)
        #kmed_idx_, tot_cost_  = k_medoids_(pca, l, labels,k=k)

        fp = kmedoids.fastpam1(euclidean_distances(pca[label_idx]), k, 100,random_state=42)
        kmed_idx=label_idx[fp.medoids]

        for idx in kmed_idx:
            #print(idx,l, labels[idx], tot_cost)
            files_of_interest.append([pdbfiles[idx], l, labels[idx]])
            pca_of_interest.append(pca[idx])
            plt.plot(pca[idx,0], pca[idx,1], 'r*', markersize=15) #highlight the medoids on the PCA plot

    prefix=f'{outpath}/{name}-k{k}-pca_n{n_components}'
    plt.savefig(f'{prefix}-cluster.png')
    if not show_plot:
        plt.clf()
    

     #create pse file with colors that match viridis colors in cluster.png
    viridis = plt.get_cmap('viridis',len(files_of_interest))
    largest_group_num = max(files_of_interest, key=lambda x: x[1])
  #  pymol.cmd.load(files[0].replace('-self.foldseek',ext), 'Dominant')
    #pymol.cmd.hide('everything', 'all')
    data=[]
    #with open(outpath + '/' + name + "-structures_of_interest.csv", "w") as file:
    #    file.write("group, file, pca_1, pca_2\n")
    for idx, foi in enumerate(files_of_interest):
        row={}
        row['group']=foi[1]
        row['file']=foi[0]
        row['pca_1']=pca_of_interest[idx][0]
        row['pca_2']=pca_of_interest[idx][1]
        row['pca_3']=pca_of_interest[idx][2]
        row['pca_4']=pca_of_interest[idx][3]
        data.append(row)
    df_sel=pd.DataFrame(data)
    df_sel.to_csv(f"{prefix}-structures_of_interest.csv",index=False)
        
           # print(idx)
            #file.write(f"{foi[1]}, {foi[0]}, {pca_of_interest[idx][0]}, {pca_of_interest[idx][1]}\n")
            #if largest_group_num[1] == -1:
            #    color = 0
            #else:
            #    color = (foi[1] + 1) / (largest_group_num[1]+1)
            #color = viridis(color)[:3]
            #new_name = re.findall(r'(full)|(max\w+)|(rank_\d+)', foi[0])
            #new_name = str(idx)+ '_' + '_'.join([i for n in new_name for i in n if i != ''])
            #pymol.cmd.load(foi[0].replace('-self.foldseek',ext), new_name)
            #pymol.cmd.align(new_name,'Dominant')
            #color_name = 'col_'+str(foi[1])
            #pymol.cmd.set_color(color_name, color)
            #pymol.cmd.color(color_name,new_name)
            #file.write(f"{foi[1]}, {foi[0]}, {pca_of_interest[idx][0]}, {pca_of_interest[idx][1]}\n")

    #pymol.cmd.save(blind_path + '/' + pdb1_name + '-structures_of_interest.pse', 'pse')
    #pymol.cmd.delete('all')
    #pymol.cmd.reinitialize()

    #save all data with clusters
    #with open(f"{outpath}/{name}-structures_all.csv", 'w') as file:
    #    file.write("group, file, pca_1, pca_2\n")
    data=[]
    for idx, f in enumerate(pdbfiles):
        row={}
        row['group']=labels[idx]
        row['file']=f
        row['pca_1']=pca[idx,0]
        row['pca_2']=pca[idx,1]
        row['pca_3']=pca[idx,2]
        row['pca_4']=pca[idx,3]
        row['n_components']=n_components
        row['k']=k
        data.append(row)
    df_all=pd.DataFrame(data)
    df_all.to_csv(f"{prefix}-structures_all.csv",index=False)

    return pca,df_sel,df_all
import argparse



def main():
    parser = argparse.ArgumentParser(description="Perform PCA, HDBSCAN, amd K-medoids clustering")
    
    # Required positional arguments
    parser.add_argument('model_path', type=str, help='Path to the model directory will search for all pdbs recursively')
    parser.add_argument('outpath', type=str, help='Output path')
    
    # Optional positional argument with default
    parser.add_argument('name', type=str, nargs='?', default='default2', 
                        help='Optional name (default: "default")')
    
     # Optional flags
    parser.add_argument('-reference_file', type=str, default=None,
                        help='.txt file with list of references)')
    parser.add_argument('-outlier_file', type=str, default='None',
                        help='outlier file to use)')
    parser.add_argument('--ext', type=str, default='.pdb',
                        help='File extension to filter (default: .pdb)')
    parser.add_argument('--show_plot', action='store_true', default=False,
                        help='Show plot (default: False).')
    parser.add_argument('-k', type=int, default=3,
                        help='Number of clusters or groups (default: 3)')
    parser.add_argument('-n_cpu', type=int, default=32,
                        help='Number of cpus for foldseek (default: 32)')
    args = parser.parse_args()

    os.makedirs(args.outpath,exist_ok=True)

    pdbfiles = glob.glob(f"{args.model_path}/**/*{args.ext}", recursive=True)
    print(f'Found {len(pdbfiles)} {args.ext} files')
    #results_data=run_all_foldseek(pdbfiles, args.outpath,n_cpu=args.n_cpu)
    #print(results_data.keys())
    
    outliers_loop=[]
    outliers=[]
    outlier_file_loops=f'{args.outpath}/outlier.loops'
    if os.path.exists(outlier_file_loops):
        print(f'Reading outliers from {outlier_file_loops}')
        with open(outlier_file_loops,'r') as f:
            outliers_loop=[line.rstrip() for line in f.readlines()]
    else:   
        outliers_loop=get_loop_outliers(pdbfiles)
        with open(outlier_file_loops, 'w') as f:
            for outlier in outliers:
                f.write(f"{outlier}\n")
    print(f'Found {len(outliers)} loop outliers.')
    
    if os.path.exists(args.outlier_file):
        print(f'Reading outliers from {args.outlier_file}')
        with open(args.outlier_file,'r') as f:
            outliers=[line.rstrip() for line in f.readlines()]
    else:
        print('Using loop outlier, since it was default')
        outliers=outliers_loop
    
    print(f'Found {len(outliers)} outliers total.')

   #if not os.path.exits(f'{args.outpath}/foldseek.done'):
    results_data=run_all_foldseek(pdbfiles, args.outpath,n_cpu=args.n_cpu)
    #    open(f'{args.outpath}/foldseek.done','w').close()

    #corr_mtx_file=f'{args.outpath}/corr_mtx.{len(outliers)}.pkl'
    #if os.path.exists(corr_mtx_file):
    #    print(f'Loading corr_mtx from file: {corr_mtx_file}')
    #    norm_corr_mtx=pickle.load(open(corr_mtx_file,'rb'))
    #else:    
    corr_mtx, foldseek_keys = populate_corr_mtx(results_data, outliers=outliers)
    print('corr_mtx', corr_mtx[:2])
    norm_corr_mtx, scaler_ = scale_norm(corr_mtx, foldseek_keys, scaler=None)
    print('corr_mtx', norm_corr_mtx['mtx'][:2])
    

    #    with open(corr_mtx_file,'wb') as f:
    #        pickle.dump(norm_corr_mtx,f)

    # MAP REFERENCES
    if args.reference_file:
        with open(args.reference_file, "r") as f:
            representatives_hits = [line.strip() for line in f if line.strip()]
            
        results_data_refs = run_all_foldseek(representatives_hits, args.outpath, n_cpu=args.n_cpu, references=True)
        corr_mtx_refs, foldseek_keys_refs = populate_corr_mtx(results_data_refs, outliers=outliers,references=True, foldseek_keys_n=foldseek_keys)
        print('corr_mtx', corr_mtx_refs[:2])
        norm_corr_mtx_refs, _ = scale_norm(corr_mtx_refs, foldseek_keys_refs, scaler=scaler_)
        print('corr_mtx', norm_corr_mtx_refs['mtx'][:2])
        
        obj = Clustering(args.outpath, args.name, 3, 4, norm_corr_mtx, norm_corr_mtx_refs)
        obj.main(map_references=True)
    
    #results_data_refs = run_all_foldseek(representatives_hits['1AD5'], args.outpath, n_cpu=args.n_cpu, references=True)
    #corr_mtx_refs, foldseek_keys_refs = populate_corr_mtx(results_data_refs, outliers=outliers,references=True, foldseek_keys_n=foldseek_keys)
    #norm_corr_mtx_refs, _ = scale_norm(corr_mtx_refs, foldseek_keys_refs, scaler=scaler_)
    #print(norm_corr_mtx_refs)

    #pca,df_sel,df_all=pca_and_cluster(args.outpath,args.name,norm_corr_mtx,n_components=4,k=args.k,show_plot=args.show_plot)
    else:
        obj = Clustering(args.outpath, args.name, 3, 4, norm_corr_mtx)
        obj.main(map_references=False)
        # df_sel,df_all=pca_and_cluster(args.outpath,args.name,norm_corr_mtx,n_components=4,k=args.k,show_plot=args.show_plot)

    # for rep in representatives_hits['1AD5']:
    #     outfile = run_foldseek(rep, db_directory=args.outpath+'/pdbs_for_db/',ext='.pdb',outpath=args.outpath)
    #     print(outfile)

if __name__ == "__main__":
    main()

'''
# EXAMPLE RUN
python cluster_models.py /proj/wallner-b/users/x_yogka/AFsample3/af3-dev/notebooks/casestudy/1AD5/allmodels 
                         local_data/1AD5 
                         -reference_file /proj/wallner-b/users/x_yogka/AFsample3/af3-dev/notebooks/representatives_hits_files/1AD5.txt
'''