import tensorflow as tf
import numpy
from mmappickle import mmapdict
from mmappickle.stubs import EmptyNDArray
import itertools

class MmapSpectralData:
    def __init__(self, filename, readonly=True):
        self._mm = mmapdict(filename, readonly=readonly)

    @property
    def balanced_length(self):
        return len(self.labels) * min(len(self.data_for_label(label)) for label in self.labels)

    @property
    def wavelengths(self):
        if 'wavelengths' not in self._mm:
            return None
        return numpy.array(self._mm['wavelengths'])

    @property
    def labels(self):
        if 'labels' not in self._mm:
            return []
        return list(self._mm['labels'])

    def data_for_label(self, label):
        if type(label) != int:
            label = self.labels.index(label)

        return self._mm['data-{:05d}'.format(label)]


    def add_data(self, mhdr_file, label):
        d = mmapdict(mhdr_file, True)

        data = d['hdr']
        mask = d['hdr'].mask.all(2)

        valid_pixels = numpy.array(numpy.nonzero(~(d['hdr'].mask.all(2)))).T

        if label not in self.labels:
            self._mm['labels'] = self.labels + [label]

        if self.wavelengths is None:
            self._mm['wavelengths'] = numpy.array(d['wavelengths'])
        else:
            if list(self.wavelengths) != list(d['wavelengths']):
                raise ValueError("Wavelengths don't match")

        label_id = self.labels.index(label)
        label_key = 'data-{:05d}'.format(label_id)

        if label_key not in self._mm:
            self._mm[label_key] = EmptyNDArray((valid_pixels.shape[0], len(self._mm['wavelengths'])), dtype=numpy.float32)
            offset = 0
        else:
            old = self._mm[label_key]
            self._mm[label_key] = EmptyNDArray((valid_pixels.shape[0]+old.shape[0], len(self._mm['wavelengths'])), dtype=numpy.float32)
            self._mm[label_key][:old.shape[0]] = old
            offset = old.shape[0]

        out_data = self._mm[label_key]
        for y, x in valid_pixels:
            out_data[offset] = data[y, x].filled(numpy.nan)
            offset += 1

        self._mm.vacuum()

    def generate_data(self, batch_size, balanced=True):
        if not balanced:
            spectrum_list = []

            for label_id, _ in enumerate(self.labels):
                label_data = numpy.zeros((self.data_for_label(label_id).shape[0], 2+len(self.labels)), dtype=numpy.int32)
                label_data[:, 0] = label_id
                label_data[:, 1] = numpy.arange(self.data_for_label(label_id).shape[0])
                label_data[:, label_id+2] = 1
                spectrum_list.append(label_data)
            spectrum_list = numpy.concatenate(spectrum_list, 0)

            while True:
                idxs = numpy.random.randint(0, len(spectrum_list), batch_size)

                batch = spectrum_list[idxs]
                batch = batch[batch[:,0].argsort()]

                spectra = numpy.concatenate([self.data_for_label(label_id)[batch[batch[:,0]==label_id,1]] for label_id, _ in enumerate(self.labels)], 0)
                one_hots = batch[:, 2:]

                yield spectra, one_hots

        else:
            #balanced
            spectrum_list = []
            spectrum_dict = {}
            spectrum_base_idx = 0
            for label_id, _ in enumerate(self.labels):
                label_data = numpy.zeros((self.data_for_label(label_id).shape[0], 2+len(self.labels)), dtype=numpy.int32)
                label_data[:, 0] = label_id
                label_data[:, 1] = numpy.arange(self.data_for_label(label_id).shape[0])
                label_data[:, label_id+2] = 1
                spectrum_dict[label_id] = numpy.arange(spectrum_base_idx, spectrum_base_idx+label_data.shape[0])
                spectrum_list.append(label_data)
                spectrum_base_idx += label_data.shape[0]

            spectrum_list = numpy.concatenate(spectrum_list, 0)

            while True:
                subbatch_size = batch_size // len(self.labels)
                idxs = []
                for k, v in spectrum_dict.items():
                    idxs += list(v[numpy.random.randint(0, len(v), subbatch_size)])

                idxs += list(numpy.random.randint(0, len(spectrum_list), batch_size-len(idxs)))

                batch = spectrum_list[idxs]
                batch = batch[batch[:,0].argsort()]

                spectra = numpy.concatenate([self.data_for_label(label_id)[batch[batch[:,0]==label_id,1]] for label_id, _ in enumerate(self.labels)], 0)
                one_hots = batch[:, 2:]

                yield spectra, one_hots


    def generate_data_reverse(self, *a, **kw):
        for spectra, one_hots in self.generate_data(*a, **kw):
            yield one_hots, spectra


    def generate_data_filled0(self, *a, **kw):
        for spectra, one_hots in self.generate_data(*a, **kw):
            yield numpy.ma.masked_invalid(spectra).filled(0), one_hots

    def generate_data_filled_and_mask(self, mask_probability=0.1, *a, **kw):
        for spectra, one_hots in self.generate_data(*a, **kw):
            spectra_masked = numpy.ma.masked_invalid(spectra)
            spectra_masked[numpy.random.random(spectra_masked.shape)<mask_probability] = numpy.ma.masked

            interleaved = numpy.zeros((spectra_masked.shape[0], spectra_masked.shape[1]*2))
            interleaved[:, ::2] = spectra_masked.filled(0)
            interleaved[:, 1::2] = spectra_masked.mask

            yield interleaved, one_hots

    def generate_data_filled_normalized_and_mask(self, mask_probability=0.1, *a, **kw):
        for spectra, one_hots in self.generate_data(*a, **kw):
            spectra_masked = numpy.ma.masked_invalid(spectra)
            spectra_masked[numpy.random.random(spectra_masked.shape)<mask_probability] = numpy.ma.masked

            spectra_masked = (spectra_masked - spectra_masked.mean(1, keepdims=True)) / spectra_masked.std(1, keepdims=True)

            interleaved = numpy.zeros((spectra_masked.shape[0], spectra_masked.shape[1]*2))
            interleaved[:, ::2] = spectra_masked.filled(0)
            interleaved[:, 1::2] = -(spectra_masked.mask.astype(numpy.float)) + 0.5

            yield interleaved, one_hots

    def generate_data_filled_normalized(self, mask_probability=0.1, *a, **kw):
        for spectra, one_hots in self.generate_data(*a, **kw):
            spectra_masked = numpy.ma.masked_invalid(spectra)
            spectra_masked[numpy.random.random(spectra_masked.shape)<mask_probability] = numpy.ma.masked

            spectra_masked = (spectra_masked - spectra_masked.mean(1, keepdims=True)) / spectra_masked.std(1, keepdims=True)

            yield spectra_masked.filled(0), one_hots


if __name__ == '__main__':
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset')
    parser.add_argument('--add', nargs='+', help='add data using files')

    args = parser.parse_args()

    mmsd = MmapSpectralData(args.dataset, False)

    if args.add is not None:
        for f in args.add:
            print(f)
            descrfile = os.path.join(os.path.dirname(f), 'description.txt')
            if not os.path.exists(descrfile):
                continue

            descrdata = [x.strip() for x in open(descrfile).read().strip().split('\n')]
            descrdata = [x.split(':',1) for x in descrdata]
            descrdata = dict([(x.strip().lower(), y.strip()) for x,y in descrdata])

            mineral_name = descrdata['name'].lower()
            mmsd.add_data(f, mineral_name)

    for l_id, l in enumerate(mmsd.labels):
        print(l, mmsd.data_for_label(l_id).shape)
