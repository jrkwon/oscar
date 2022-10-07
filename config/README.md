# System Configuration

## conda
- `environment.yaml`: conda environment file


```
$ conda env create -f conda/environment.yaml
```

## config.yaml
Specify yaml file names for each system setting.
- `neural_net`
- `data_collection`
- `run_neural`

### Naming Convention
Use `<vehicle>-<user>-<track>.yaml` for a yaml filename. 

For example, `funsion_jaerock_jaerock2` means that vehicle `fusion` is being used by user `jaerock` on track `jaerock2`.

## data_collection
- `fusion_template.yaml`: configuration example of fusion for data_collection package
- `rover_template.yaml`: configuration example of rover for data_collection package

## neural_net
- `fusion_template.yaml`: configuration example of fusion for neural_net module
- `rover_template.yaml`: configuration example of rover for neural_net module
 
## run_neural
- `fusion_template.yaml`: configuration example of fusion for run_neural package
- `rover_template.yaml`: configuration example of rover for run_neural package
