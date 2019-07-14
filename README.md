## Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

This is a PyTorch implementation of the [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf).

## Dataset

You can find CycleGAN datasets from [this](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets), and chose what kind of dataset you want to download.

To train a model on your own datasets, you need to create a data folder as the following (apple2orange is dataset name).

```
apple2orange/
	train/
		X/...jpg
		Y/...jpg
	test/
		X/...jpg
		Y/...jpg
```

## Usage

`python cyclegan.py ARGS`

Possible ARGS are:

- `--img_size` size of image, default is `128`;
- `--batch_size` size of minibatch, default is `8`;
- `--dataset_name` name of the dataset, default is `'apple2orange'`;
- `--n_residual_blocks` number of residual blocks in the generator, default is `6`;
- `--max_epoch` numbers of epoch to train, default is `200`;
- `--lr` the learning rate of Adam, default is `0.0002`;
- `--decay_start_epoch` start of epoch to decay lr, default is `100`;
- `--sample_interval` interval between saving generator outputs, default is `100`;
- `--checkpoint_interval` interval between saving models, default is `100`;
- `--save_images` directory of saving generator outputs, default is `'images'`;
- `--save_models` directory of saving models, default is `'save_models'`.

An example:

```
python cyclegan.py --dataset_name "apple2orange"
```

## Result



