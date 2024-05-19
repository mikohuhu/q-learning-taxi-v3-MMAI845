# this is a set of codes with custom_taxi_env.py
added one Extra passenger location (2, 2): self.passenger_locations = [(0, 0), (0, 4), (4, 0), (4, 3), (2, 2)]  

wall numbers I'm not sure how many in original environment, but in this custom_taxi_env.py only 3 walls:  self.walls = [(1, 2), (2, 2), (3, 2)]  # Example walls

execution order will be the same

## Instructions to run

```shell script
$ pip install -r requirements.txt
```

### Training
```shell script
$ python train.py --help
Usage: train.py [OPTIONS]

Options:
  --num-episodes INTEGER  Number of episodes to train on  [default: 100000]
  --save-path TEXT        Path to save the Q-table dump  [default:
                          q_table.pickle]
  --help                  Show this message and exit.
```

### Evaluation

```shell script
$ python evaluate.py --help
Usage: evaluate.py [OPTIONS]

Options:
  --num-episodes INTEGER  Number of episodes to train on  [default: 100]
  --q-path TEXT           Path to read the q-table values from  [default:
                          q_table.pickle]
  --help                  Show this message and exit.
```

