if __name__ == '__main__':
    import argparse
    import numpy
    import os
    #Disable TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    from matplotlib import pyplot as plt

    from .customlayers import *
    from .dataset import MmapSpectralData
    custom_keras_objects = {
        'HIILLayer': HIILLayer,
        'NanDropout': NanDropout,
        'NanFeatureScalingNormalizationLayer': NanFeatureScalingNormalizationLayer,
        'FeatureScalingNormalizationLayer': FeatureScalingNormalizationLayer,
        'LabelDenseLayer': LabelDenseLayer,
    }

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help', dest="command")
    subparsers.required = True


    parser_dataset_create = subparsers.add_parser('dataset_create', help='create a dataset')
    parser_dataset_create.add_argument('--dataset', required=True, help='dataset')
    parser_dataset_create.add_argument('--add', nargs='+', help='add data using files')

    def subdataset_spec(s):
        try:

            x, y = s.split(',', 1)
            if x == '*':
                return numpy.inf, y
            return float(x), y
        except:
            raise argparse.ArgumentTypeError("Subdataset specification must be percent,filename")

    parser_dataset_split = subparsers.add_parser('dataset_split', help='split a dataset')
    parser_dataset_split.add_argument('--dataset', required=True, help='input dataset')
    parser_dataset_split.add_argument('--parts', nargs='+', required=True, type=subdataset_spec, help='output datasets (perc,filename)')
    parser_dataset_split.add_argument('--onlyfull', action='store_true', help='only spectra without holes')

    parser_dataset_pokeholes = subparsers.add_parser('dataset_pokeholes', help='Poke holes in a dataset')
    parser_dataset_pokeholes.add_argument('--input', required=True, help='input dataset')
    parser_dataset_pokeholes.add_argument('--output', required=True, help='output dataset')
    parser_dataset_pokeholes.add_argument('--holes', nargs='+', type=float, help='hole percentage')


    def layer_spec(s):
        try:

            x, y = s.split(',', 1)
            return int(x), float(y)
        except:
            raise argparse.ArgumentTypeError("Layer specification must be width,dropout")

    parser_model_create = subparsers.add_parser('model_create', help='create a model')
    parser_model_create.add_argument('--model', help='model to create', required=True)
    parser_model_create.add_argument('--template', help='dataset to use as a template', required=True)
    parser_model_create.add_argument('--input_layer', help='input layer specification', required=True)
    parser_model_create.add_argument('--input_dropout', help='dropout [0-1] after the input layer [0.5]', default=0.5, type=float)
    parser_model_create.add_argument('--layers', nargs='+', type=layer_spec, help='layer specification [width, dropout]')

    parser_model_train = subparsers.add_parser('model_train', help='train a model')
    parser_model_train.add_argument('--model', help='model', required=True)
    parser_model_train.add_argument('--ds_train', help='train dataset', required=True)
    parser_model_train.add_argument('--ds_validation', help='validation dataset')
    parser_model_train.add_argument('--epochs', help='number of epochs [1]', default=1, type=int)
    parser_model_train.add_argument('--batch_size', help='batch size [10000]', default=10000, type=int)
    parser_model_train.add_argument('--log', help='log file')
    parser_model_train_output = parser_model_train.add_mutually_exclusive_group(required=True)
    parser_model_train_output.add_argument('--output', help='model output')
    parser_model_train_output.add_argument('--inplace', help='model output to the --model file', action='store_true')


    parser_model_test = subparsers.add_parser('model_test', help='test a model')
    parser_model_test.add_argument('--model', help='model', required=True)
    parser_model_test.add_argument('--dataset', help='dataset', required=True)
    parser_model_test_fill = parser_model_test.add_mutually_exclusive_group()
    parser_model_test_fill.add_argument('--zerofill', help='Fill NaN with 0', action='store_true')
    parser_model_test_fill.add_argument('--interpfill', help='Fill NaN by linear interpolation', action='store_true')

    parser_model_test_interactive = subparsers.add_parser('model_test_interactive', help='test a model')
    parser_model_test_interactive.add_argument('--model', help='model', required=True)
    parser_model_test_interactive.add_argument('--dataset', help='dataset', required=False)

    parser_model_create_recoplot = subparsers.add_parser('model_create_recoplot', help='create a recognition plot for a model')
    parser_model_create_recoplot.add_argument('--model', help='model', required=True)
    parser_model_create_recoplot.add_argument('--dataset', help='dataset', required=True)
    parser_model_create_recoplot.add_argument('--label', help='label', required=True)
    parser_model_create_recoplot.add_argument('--output', help='output')

    args = parser.parse_args()

    if args.command == 'dataset_create':
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

    elif args.command == 'dataset_split':
        from mmappickle import mmapdict
        input_dataset = mmapdict(args.dataset, True)

        datakeys = [k for k in input_dataset.keys() if k.startswith('data-')]
        datarem = {}
        datacount = {}

        for k in datakeys:
            if args.onlyfull:
                datarem[k] = numpy.nonzero(numpy.isnan(input_dataset[k]).sum(1)==0)[0]
            else:
                datarem[k] = numpy.arange(input_dataset[k].shape[0])

            datacount[k] = len(datarem[k])



        for p, output_dataset_name in args.parts:
            output_dataset = mmapdict(output_dataset_name)
            output_dataset['labels'] = input_dataset['labels']
            output_dataset['wavelengths'] = input_dataset['wavelengths']

            for k in datakeys:
                if p == numpy.inf:
                    output_dataset[k] = input_dataset[k][datarem[k]]
                    datarem[k] = numpy.array([])
                else:
                    data_for_key = numpy.random.choice(datarem[k], int(p*datacount[k]))
                    datarem[k] = numpy.array(list(set(datarem[k]).difference(data_for_key)))
                    output_dataset[k] = input_dataset[k][data_for_key]

    elif args.command == 'dataset_pokeholes':
        from mmappickle import mmapdict
        input_dataset = mmapdict(args.input, True)
        output_dataset = mmapdict(args.output, False)

        for k in input_dataset.keys():
            output_dataset[k] = input_dataset[k]

        datakeys = [k for k in input_dataset.keys() if k.startswith('data-')]

        for hole in args.holes:
            for k in datakeys:
                hole_width = int(hole * output_dataset[k].shape[1])

                hole_start = numpy.random.randint(output_dataset[k].shape[1]-hole_width, size=output_dataset[k].shape[0])
                for rowid, hs in enumerate(hole_start):
                    output_dataset[k][rowid, hs:hs + hole_width] = numpy.nan


    elif args.command == 'model_create':
        mmsd = MmapSpectralData(args.template)

        # create model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Reshape((len(mmsd.wavelengths), ), input_shape=(len(mmsd.wavelengths), )))
        input_layer_spec = args.input_layer.split(',')
        if input_layer_spec[0] == 'HIILLayer':
            minwl, maxwl, depth = [int(x) for x in input_layer_spec[1:]]
            model.add(HIILLayer(mmsd.wavelengths, minwl, maxwl, depth))
        elif input_layer_spec[0] == 'Identity':
            pass
        else:
            model.add(custom_keras_objects[input_layer_spec[0]]())

        model.add(tf.keras.layers.Dropout(args.input_dropout))

        if args.layers is not None:
            for w, d in args.layers:
                model.add(tf.keras.layers.Dense(w, activation='relu'))
                model.add(tf.keras.layers.Dropout(d))

        model.add(LabelDenseLayer(mmsd.labels, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.save(args.model)
        model.summary()

    elif args.command == 'model_train':
        model = tf.keras.models.load_model(args.model, custom_objects=custom_keras_objects)
        ds_train = MmapSpectralData(args.ds_train)
        if args.ds_validation:
            ds_validation = MmapSpectralData(args.ds_validation)
            r = model.fit_generator(
                ds_train.generate_data(batch_size=args.batch_size, balanced=True),
                steps_per_epoch=int(round(ds_train.balanced_length/args.batch_size)),
                validation_data=ds_validation.generate_data(batch_size=args.batch_size, balanced=True),
                validation_steps=int(round(ds_validation.balanced_length/args.batch_size)),
                epochs=args.epochs
            )
        else:
            r = model.fit_generator(
            ds_train.generate_data(batch_size=args.batch_size, balanced=True),
                steps_per_epoch=int(round(ds_train.balanced_length/args.batch_size)),
                epochs=args.epochs
            )

        if args.log:
            import pickle
            pickle.dump(r.history, open(args.log, 'wb'))

        if args.inplace:
            model.save(args.model)
        else:
            model.save(args.output)

    elif args.command == 'model_test':
        model = tf.keras.models.load_model(args.model, custom_objects=custom_keras_objects)
        dataset = MmapSpectralData(args.dataset)

        labels_model = set(model.layers[-1].labels)
        labels_dataset = set(dataset.labels)

        labels = list(sorted(labels_model.intersection(labels_dataset)))

        predmatrix = numpy.zeros((len(labels), len(labels)))
        predmatrix_n = numpy.zeros((len(labels), len(labels)))
        for label_id, label_name in enumerate(labels):
            data = dataset.data_for_label(label_name)
            if args.zerofill:
                data = numpy.ma.masked_invalid(data).filled(0)
            elif args.interpfill:
                data = data.copy()
                holes = set(tuple(x) for x in numpy.array(numpy.nonzero(numpy.isnan(data))).T)
                while len(holes) > 0:
                    hole = holes.pop()
                    row = data[hole[0]]
                    pos_lower = pos_higher = hole[1]
                    while pos_lower > 0 and numpy.isnan(row[pos_lower]):
                        pos_lower -= 1

                    while pos_higher < len(row) - 1 and numpy.isnan(row[pos_higher]):
                        pos_higher += 1

                    if numpy.isnan(row[pos_lower]) and numpy.isnan(row[pos_higher]):
                        row[:] = 0
                    elif numpy.isnan(row[pos_lower]):
                        row[pos_lower:pos_higher+1] = row[pos_higher]
                    elif numpy.isnan(row[pos_higher]):
                        row[pos_lower:pos_higher+1] = row[pos_lower]
                    else:
                        lsp = numpy.linspace(0, 1, pos_higher-pos_lower+1)
                        row[pos_lower:pos_higher+1] = lsp * row[pos_higher] + (1 - lsp) * row[pos_lower]

                    for p in range(pos_lower, pos_higher+1):
                        holes.discard((hole[0], p))

            output = model.predict(data)
            predmatrix[label_id] = [(output.argmax(1) == model.layers[-1].labels.index(ln)).sum() for ln in labels]
            predmatrix_n[label_id] = predmatrix[label_id] / predmatrix[label_id].sum()
            #print(label_id, label_name, predmatrix_n[label_id])

        print(numpy.diag(predmatrix).sum() / predmatrix.sum())

    elif args.command == 'model_test_interactive':
        model = tf.keras.models.load_model(args.model, custom_objects=custom_keras_objects)
        labels_model = set(model.layers[-1].labels)

        if args.dataset:
            dataset = MmapSpectralData(args.dataset)
            labels_dataset = set(dataset.labels)
            labels = list(sorted(labels_model.intersection(labels_dataset)))
        else:
            labels = list(sorted(labels_model))

        import IPython
        IPython.embed()

    elif args.command == 'model_create_recoplot':
        model = tf.keras.models.load_model(args.model, custom_objects=custom_keras_objects)
        dataset = MmapSpectralData(args.dataset)

        labels_model = set(model.layers[-1].labels)
        labels_dataset = set(dataset.labels)

        labels = list(sorted(labels_model.intersection(labels_dataset)))

        data = numpy.ma.masked_invalid(dataset.data_for_label(args.label)[::20]).filled(0)
        data_tmp = numpy.empty_like(data)
        output = numpy.zeros((256,256))*numpy.nan
        for l in range(0,255,1):
            for o in range(0,256-l,1):
                data_tmp[:,:] = numpy.nan
                data_tmp[:,o:o+l] = data[:,o:o+l]

                pred_correct = numpy.argmax(model.predict(data_tmp),1) == model.layers[-1].labels.index(args.label)
                output[l,o+l//2] = pred_correct.mean()
                print(o,l,pred_correct.mean())

        refldata = dataset.data_for_label(args.label).copy()
        refldata = (refldata - refldata.mean(1, keepdims=True)) / refldata.std(1, keepdims=True)
        refldata -= refldata.min()
        refldata /= refldata.max()

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), dpi=300)

        refldata = dataset.data_for_label(args.label).copy()
        refldata = (refldata - refldata.mean(1, keepdims=True)) / refldata.std(1, keepdims=True)
        refldata -= numpy.percentile(refldata, 0.1)
        refldata /= numpy.percentile(refldata,99.9)

        r = numpy.histogram2d(numpy.tile(dataset.wavelengths,refldata.shape[0]),refldata.flatten(), [dataset.wavelengths-0.01, numpy.linspace(0, 1, 100)])
        axes.flat[0].imshow(r[0].T[::-1], extent=[dataset.wavelengths.min(), dataset.wavelengths.max(), 1, 0], aspect='auto', cmap='gray_r')
        axes.flat[0].set_xlim(dataset.wavelengths.min(), dataset.wavelengths.max())
        axes.flat[0].set_xlabel("Wavelength [nm]")
        axes.flat[0].set_ylabel("Normalized reflectance [-]")
        axes.flat[0].set_yticks([])

        im = axes.flat[1].imshow(output, vmin=0, vmax=1, aspect='auto')
        axes.flat[1].set_xlabel("Band number")
        axes.flat[1].set_ylabel("Width")

        plt.tight_layout()

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        if args.output:
            plt.savefig(args.output)
        else:
            plt.show()
