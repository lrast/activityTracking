# methods for writing to files

import torch
import safetensors.torch

from torch.utils.data import Dataset
from safetensors import safe_open

from monitoring import activate_model_tracking, clear_hooks


class ActivityDataset(Dataset):
    """ Dataset of tensor activity """
    def __init__(self, directory, device='cpu'):
        super(ActivityDataset, self).__init__()
        directory = directory.rstrip('/')

        metadata = torch.load(f'{directory}/meta.data', weights_only=True)
        self.len = metadata[0]
        self.number_per_file = metadata[1]
        self.layer_names = metadata[2]

        files_max = metadata[0] // self.number_per_file + 1

        self.files = [[safe_open(f'{directory}/{layer}-{number}.st',
                                 framework="pt", device=device)
                       for layer in self.layer_names]
                      for number in range(files_max)]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fnum = index // self.number_per_file
        ind = index % self.number_per_file

        return tuple(file.get_tensor(str(ind)) for file in self.files[fnum])


class ActivityRecorder(object):
    """ ActivityRecorder: records neural network activity
    """
    def __init__(self, model, layers_to_track, directory,
                 track_inputs=False, num_per_file=1000):
        super(ActivityRecorder, self).__init__()
        self.num_per_file = num_per_file
        self.track_inputs = track_inputs
        self.directory = directory.rstrip('/')

        if track_inputs:
            self.layer_names = [name + '_input' for name in layers_to_track] + \
                               [name + '_output' for name in layers_to_track]
        else:
            self.layer_names = layers_to_track

        self.buffers = {name: BufferToFile(name, self.directory, num_per_file)
                        for name in self.layer_names}

        self.model = model
        self.active_hooks = activate_model_tracking(model, layers_to_track,
                                                    self.ingest,
                                                    track_inputs=track_inputs)

    def ingest(self, name, outputs, inputs=None):
        """ Callback for hooks to send their data to """
        if self.track_inputs:
            self.buffers[name + '_output'].batch_append(outputs.detach().clone())
            self.buffers[name + '_input'].batch_append(inputs.detach().clone())

        else:
            self.buffers[name].batch_append(outputs.detach())

    def close(self):
        totals_seen = [buffer.total_entries for buffer in self.buffers.values()]
        assert all(map(lambda x: x == totals_seen[0], totals_seen))

        metadata = [totals_seen[0], self.num_per_file, self.layer_names]
        torch.save(metadata, f'{self.directory}/meta.data')

        for buffer in self.buffers.values():
            buffer.close()

        clear_hooks(self.active_hooks)

    def load(self):
        """ load partially completed activity records """
        raise NotImplementedError()


class BufferToFile(list):
    """ BufferToFile: writes to a file after a certain number of entries """
    def __init__(self, name, directory, max_len=1000):
        super(BufferToFile, self).__init__()
        self.name = name
        self.directory = directory
        self.total_entries = 0
        self.max_len = max_len

    def append(self, item):
        """ Adds new item to buffer """
        super(BufferToFile, self).append(item)
        self.total_entries += 1

        if len(self) >= self.max_len:
            self.save_contents_to_file()

    def batch_append(self, batch):
        """ extends by a batch (along the first dimension) """
        for tensor in batch:
            self.append(tensor)

    def save_contents_to_file(self):
        """ write buffer contents to a file """
        file_num = (self.total_entries - 1) // self.max_len

        dict_version = {str(i): v for (i, v) in enumerate(self)}
        safetensors.torch.save_file(dict_version,
                                    f'{self.directory}/{self.name}-{file_num}.st')

        self.clear()

    def close(self):
        """ close buffer """
        self.save_contents_to_file()

    def load(self):
        raise NotImplementedError()
