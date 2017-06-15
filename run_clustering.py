import random
import pickle
import re
import os

import numpy as np
import matplotlib.pyplot as plt
import faiss

import ecg
from ecg.utils import tools
from ecg.utils.diseases import all_holter_diseases


def plot_ecg(channels,  samples_per_pic, save_path, file_name,
            first_sample=None, last_sample=None, channel_names=['ES', 'AS', 'AI'],
            convert_from_easi=True, beats=None, events=None, diseases=None,
            predicted_diseases=None, predicted_events=None, predicted_beats=None,
            probabilities=None, verbose=False, not_plot_good_chunks=False,
            max_failed_plots=100):

    """Plots channels with provided beats and/or events. Saves figures to 
        save_path/file_name/

    Args:
        channels: list on channels to plot.
        first_sample: sample number to start plotting from. If not specified, 
            defaults to the first sample of the current file.
        last_sample: last sample to plot. If not specified, defaults to
            the last sample of the current file.
        samples_per_pic: number of sample per picture, 2-4k is recommended
        save_path: folder to save pics in. Will be created automatically.
        file_name: name of file.
        beats: array of beats positions is samples starting from 0.
        events: events if form of numpy array of shape (?, n_diseases). Each
            event should correspond to a beat.
        diseases: list of diseases. Length should be the same as second dimension
            of events array.
        predicted_events: same as events.
        predicted_beats: same as beats.
        probabilities: probabilities of predicted events in norm on numpy array
            of shape (?, n_diseases).
        verbose: whether to print progress.
        not_plot_good_chunks: whether or not to plot chunks with correct predictions.
            Require both `events` and `predicted events`. 
        max_failed_plots: maximum number of figures with incorrect predictions.
            Used only when `not_plot_good_chunks` is True.

    Raises:
        ValueError: if some arrays have inconsistent dimensions
             (e.g. events don't correspond to beats).
    """

    if verbose:
        print('Plotting {}.'.format(file_name))

    if events is not None:
        if events.shape[1] != len(diseases):
            raise ValueError('Each element of event should correspond to a particular disease.')
        if beats is None or (len(events) != len(beats)):
            raise ValueError('Events could not be plotted without corresponding beats.')
        if predicted_events is not None:
            if len(predicted_events) != len(events):
                raise ValueError('Each predicted event should correspond to a true event.')
            if predicted_events.shape[1] != len(predicted_diseases):
                raise ValueError('Each element of predicted event should correspond to a particular disease.')
        if probabilities is not None:
            if len(probabilities) != len(events):
                raise ValueError('Each probability of event should correspond to a true event.')

    tools.maybe_create_dirs(os.path.join(save_path, file_name))
    if channel_names != ['ES', 'AS', 'AI']:
        if convert_from_easi:
            channels = tools.convert_channels_from_easi(channels, channel_names)

    channel_length = len(channels[0])
    if last_sample is None or last_sample > channel_length:
        last_sample = channel_length

    if first_sample is None:
        first_sample = 0

    start = first_sample
    end = start + samples_per_pic

    if not_plot_good_chunks:
        if all(arg is not None for arg in [events, predicted_events, beats]):
            fails = beats[np.not_equal(events, predicted_events).any(axis=1)]
            print('{n_fails} fails total.'.format(n_fails=len(fails)))
            if len(fails) > max_failed_plots:
                fails = np.random.choice(fails, max_failed_plots, replace=False)
        else:
            raise ValueError('To skip good chunks, `events`,' \
                ' `predicted_events` and `beats` should be provided.')

    plot = True
    for _ in range(((last_sample-first_sample)//samples_per_pic)):

        xlims = (start, end)
        ylims = (-0.6, 0.6)

        if not_plot_good_chunks:
            curr_samples = np.arange(start, end)
            intersect = np.intersect1d(curr_samples, fails, assume_unique=True)
            plot = len(intersect) > 0

        if plot:
            if beats is not None:
                idx = (beats >= start) & (beats <= end)
                current_beats = beats[idx]

            if predicted_beats is not None:
                idx = (predicted_beats >= start) & (predicted_beats <= end)
                current_predicted_beats = predicted_beats[idx]
            
            t = range(start, end)
            fig = plt.figure(figsize=(25, len(channels)*5))

            channel_name = iter(channel_names)
            for i, ch in enumerate(ch[start:end] for ch in channels):
                ax = fig.add_subplot(len(channels), 1, i+1)

                ax.plot(t, ch, lw=1.0, c='b', label='original', alpha=0.3)
                ax.axhline(y=0, ls='dashed', alpha=0.1, c='g')

                if beats is not None:
                    for b in current_beats:
                        ax.axvline(b, lw=1, color='r', ls='dashdot', alpha=0.4)
                        if events is not None:
                            event = events[beats == b].ravel()
                            if event.any():
                                name = ('\n').join(np.array(diseases)[event != 0])
                                ax.text(b, 0.9*ylims[1], name, fontsize=7, color='g',
                                                     ha='center', rotation=60, alpha=0.7)       
                        if predicted_events is not None:
                            predicted_event = predicted_events[beats == b].ravel()
                            if predicted_event.any():
                                name = ('\n').join(np.array(predicted_diseases)[predicted_event != 0])
                                ax.text(b, 0.2*ylims[1], name, fontsize=7, color='m',
                                                     ha='center', rotation=60, alpha=1)     
                        if probabilities is not None:   
                            probability = probabilities[beats == b].ravel()
                            text_probs = (' ').join('{:.2f}'.format(p) for p in probability)
                            ax.text(b, -0.2*ylims[1], text_probs, fontsize=7, color='k',
                                                 ha='center', rotation=60, alpha=0.7)                                      

                if predicted_beats is not None:
                    for b in current_predicted_beats:
                        ax.axvline(b, lw=1, color='g', ls='dashed', alpha=0.4)

                ax.set_xticks(np.arange(*xlims, step=20))                                                       
                ax.set_yticks(np.arange(*ylims, step=0.1))                                                       
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.grid(which='minor', alpha=0.2)                                                
                ax.grid(which='major', alpha=0.3)      
                ax.set_ylim(ylims)
                ax.set_xlim(xlims)
                ax.set_title(next(channel_name))
                ax.legend(loc=2, prop={'size': 10})

            fig.savefig(os.path.join(save_path, '{0}_{1}.png'.format(file_name, start)), bbox_inches='tight')
            plt.close(fig)

        start += samples_per_pic
        end += samples_per_pic
        if end > last_sample:
            break

    if verbose:
        print('Finished plotting {}.'.format(file_name))

def plot_centroid(sample_index, beat_indexes, file_paths, save_path, path_to_data_files,
    neighbor):
    beat_index = beat_indexes[sample_index]

    file_path = file_paths[sample_index]
    file_path = file_path.split('_latent_state')[0]+'.npy'
    file_name = tools.get_file_name(file_path)
    file_path = path_to_data_files + file_name+'.npy'
    file_name = str(neighbor) + '_' + file_name

    data = np.load(file_path).item()
    channels = tools.get_channels(data)
    first_sample = data['beats'][beat_index-3 if beat_index-3>=0 else 0]
    last_sample = data['beats'][beat_index+3 if beat_index+3<data['beats'].shape[0] else data['beats'].shape[0]-1]
    plot_ecg(channels=channels, samples_per_pic=last_sample - first_sample,
        save_path=save_path, file_name=file_name, first_sample=first_sample,
        last_sample=last_sample, beats=data['beats'], events=data['events'],
        diseases=all_holter_diseases)

def plot_centroids(n_neighbor, I, beat_indexes, file_paths, path_to_data_files):
    for i in range(len(I)):
        print('plot cluster', i)
        for neighbor in range(n_neighbor):
            plot_centroid(I[i, neighbor], beat_indexes, file_paths,
                'Cluster_{}'.format(i), path_to_data_files, neighbor)

def plot_cluster_variance(channels,  samples_per_pic, save_path, file_name, fig_settings,
            first_sample=None, last_sample=None, channel_names=['ES', 'AS', 'AI'],
            convert_from_easi=True, beats=None, events=None, diseases=None,
            predicted_diseases=None, predicted_events=None, predicted_beats=None,
            probabilities=None, verbose=False, not_plot_good_chunks=False,
            max_failed_plots=100):

    channel_length = len(channels[0])
    call_number = fig_settings[2]

    t = np.arange(0, (last_sample-first_sample))*1000/175

    fig = fig_settings[0]
    subplots =fig_settings[1]

    channel_name = iter(channel_names)
    for i, ch in enumerate(ch[first_sample:last_sample] for ch in channels):
        ax = subplots[i]
        lw = 4-call_number*0.5
        if lw < 0.9: lw = 0.9
        ax.plot(t, ch, lw=lw, label=file_name, alpha=0.3)
        ax.set_xticks(np.arange(0, t[-1], 200))
        ax.grid(which='minor', alpha=0.2)                                                
        ax.grid(which='major', alpha=0.3)      
        ax.set_title(next(channel_name))
        ax.legend(loc=2, prop={'size': 5})



def encode_dataset(path_to_cash, paths_to_Z_with_pattern, required_diseases):
    # required_diseases: list of string with name of reuiered diseases

    def leave_required_samples(mu, beat_indexes, file_paths, y):
        inds = y[:, all_holter_diseases.index('Atrial_PAC')]>0.5
        mu = mu[inds,:].astype(np.float32)
        beat_indexes = beat_indexes[inds]
        file_paths = file_paths[inds]
        return mu, beat_indexes, file_paths

    print('encode_dataset')
    os.makedirs(os.path.dirname(path_to_cash), exist_ok=True)
    if os.path.isfile(path_to_cash):
        print('load mu, beat_indexes, file_paths')  
        with open(path_to_cash, 'rb') as f:
            mu, beat_indexes, file_paths = pickle.load(f)
    else:
        print('Can not load mu, beat_indexes, file_paths, make new ones')
        for i,path in enumerate(paths_to_Z_with_pattern):
            data = np.load(path).item()
            if i==0:
                mu, beat_indexes, file_paths = leave_required_samples(data['mu'],
                    data['beat_indexes'], np.array([path]*data['mu'].shape[0], dtype=object),
                    data['events'])
            else:
                res = leave_required_samples(data['mu'],
                    data['beat_indexes'], np.array([path]*data['mu'].shape[0], dtype=object),
                    data['events'])
                mu = np.concatenate([mu,res[0]], 0)
                beat_indexes = np.concatenate([beat_indexes,res[1]], 0)
                file_paths = np.concatenate([file_paths, res[2]], 0)
        print('Find {} PAC Z-codes'.format(len(mu)))

        with open(path_to_cash, 'wb') as f:
            pickle.dump([mu, beat_indexes, file_paths], f)
    return mu, beat_indexes, file_paths

def find_pattern(dat_files, pattern, verbose=False):
    paths = []
    for dat_file in dat_files:         
        with open(dat_file, 'r', encoding='utf-16') as f:
            lines = f.readlines()
            text = ''.join(line.rstrip() for line in lines)
        finding = re.search(pattern, text)
        if finding is not None:
            paths.append(dat_file)
    if verbose:
        print('find {} files with required pattern.'.format(len(paths)))
    return paths

def kmeans(x, ncentroids, niter, verbose):
    print('kmeans')
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter, verbose)
    kmeans.cp.max_points_per_centroid = 100000
    print('train kmeans')
    kmeans.train(x)
    return kmeans

def print_clustering_results(I, ncentroids, y, required_diseases):
    for cluster in range(ncentroids):
        print('\nCluster', cluster)

        ind = I[:,0]==cluster
        y_ = y[ind,...]
        cluster_size = np.sum(ind)
        print('Cluster_size', cluster_size)

        n_diseases_arr_cl = np.sum(y_, 0)
        n_diseases_arr_all = np.sum(y, 0)
        for i,d in enumerate(required_diseases):
            print(d+' {0} item: \t{1}% of cluster size, \t{2}% of all such labels'.format(
                int(n_diseases_arr_cl[i]),
                int(n_diseases_arr_cl[i]/cluster_size*100),
                int(n_diseases_arr_cl[i]/n_diseases_arr_all[i]*100)))

def search_files_with_pattern(paths_to_dat, paths_to_Z):
    dat_file_names = [tools.get_file_name(p) for p in paths_to_dat]
    # print(dat_file_names)
    Z_junc_paths = []
    for zp in paths_to_Z:
        for dp in dat_file_names:
            if dp in zp:
                Z_junc_paths.append(zp)
                break
    # print(Z_junc_paths)
    print('find {} Z files with specific patern'.format(len(Z_junc_paths)))
    return Z_junc_paths


def plot_variance(sample_index, beat_indexes, file_paths, save_path, path_to_data_files,
    neighbor, fig_settings):
    beat_index = beat_indexes[sample_index]

    file_path = file_paths[sample_index]
    file_path = file_path.split('_latent_state')[0]+'.npy'
    file_name = tools.get_file_name(file_path)
    file_path = path_to_data_files + file_name+'.npy'
    file_name = str(neighbor) + '_' + file_name

    data = np.load(file_path).item()
    channels = tools.get_channels(data)
    first_sample = data['beats'][beat_index]-200
    last_sample = data['beats'][beat_index]+200
    if first_sample>=0 and last_sample<len(channels[0]):
        plot_cluster_variance(channels=channels, samples_per_pic=last_sample - first_sample,
            save_path=save_path, file_name=file_name, fig_settings=fig_settings,
            first_sample=first_sample,
            last_sample=last_sample)

def plot_variances(n_neighbor, I, beat_indexes, file_paths, path_to_data_files):
    for i in range(len(I)):
        print('plot cluster', i)
        fig = plt.figure(figsize=(20, 10))
        subplots = [fig.add_subplot(3, 1 , i+1) for i in range(3)]
        call_number = 1
        fig_settings = [fig, subplots, call_number]
        for neighbor in range(n_neighbor):
            fig_settings[2] = neighbor+1
            save_path = 'Cluster_{}'.format(i)
            plot_variance(I[i, neighbor], beat_indexes, file_paths,
                save_path, path_to_data_files, neighbor, fig_settings)

        file_path = file_paths[I[i, neighbor]]
        file_path = file_path.split('_latent_state')[0]+'.npy'
        file_name = tools.get_file_name(file_path)

        fig.savefig('Cluster_{}.png'.format(i))
        plt.close(fig)

def get_paths_to_Z_with_pattern(path_to_cash, path_to_dat_files, path_to_Z):
    os.makedirs(os.path.dirname(path_to_cash), exist_ok=True)
    if os.path.isfile(path_to_cash):
        print('load paths to Z with pattern')
        with open(path_to_cash, 'rb') as f:
            paths_to_Z_with_pattern = pickle.load(f)
    else:
        print('Can not load paths to Z with pattern, search new one')
        paths_to_dat = tools.find_files(path_to_dat_files, '*.dat')
        paths_to_dat = find_pattern(paths_to_dat, junctional_pattern, True)
        paths_to_Z = tools.find_files(path_to_Z, '*.npy')
        paths_to_Z_with_pattern = search_files_with_pattern(paths_to_dat, paths_to_Z)
        with open(path_to_cash, 'wb') as f:
            pickle.dump(paths_to_Z_with_pattern, f)
    return paths_to_Z_with_pattern

    

if __name__ == '__main__':



    
    ##############################
    # PARAMETERS
    path_to_Z = '/data/Work/Nazar/ECG_encoder/predictions/latent_states_PAC/'
    # path_to_Z = '../data/latent_spaces/PVC/'
    path_to_data_files = '/data/Work/Dima/ecg_folder/convo_classifier/train_files/PAC/'
    path_to_dat_files = '/data/Work/ZHR-Files/'
    required_diseases=['Atrial_PAC']
    ncentroids = 30
    niter = 10
    verbose = True
    junctional_pattern = re.compile('АВ - блок')
    n_neighbor = 30
    ##############################

    paths_to_Z_with_pattern = get_paths_to_Z_with_pattern(path_to_cash='cash/Z_paths.pickle',
        path_to_dat_files=path_to_dat_files, path_to_Z=path_to_Z)
    
    mu, beat_indexes, file_paths = encode_dataset(
        path_to_cash='cash/mu_beat_indexes_file_paths.pickle',
        paths_to_Z_with_pattern=paths_to_Z_with_pattern,
        required_diseases=required_diseases)

    kmeans = kmeans(mu, ncentroids, niter, verbose)
    D, I = kmeans.index.search(mu, 1)
    np.save('cash/I.npy', I)
    [print('Cluster {0} size is {1}'.format(i, np.sum(I==i))) for i in range(ncentroids)]

    index = faiss.IndexFlatL2(mu.shape[1])
    index.add(mu)
    D, I = index.search(kmeans.centroids, n_neighbor)

    plot_variances(n_neighbor, I, beat_indexes, file_paths, path_to_data_files)
    plot_centroids(n_neighbor, I, beat_indexes, file_paths, path_to_data_files)


    # D, I = kmeans.index.search(mu, 1) #This will return the nearest centroid
        # for each line vector in x in I. D contains the squared L2 distances.

    # print_clustering_results(I, ncentroids, y, required_diseases)
