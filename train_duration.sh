#single8 mean average
python train.py -m throughput -roadnet single8 -green_duration 15 -cmt duration=15
python train.py -m throughput -roadnet single8 -green_duration 20 -cmt duration=20
python train.py -m throughput -roadnet single8 -green_duration 25 -cmt duration=25
python train.py -m throughput -roadnet single8 -green_duration 30 -cmt duration=30


python train.py -m throughput -roadnet single4 -green_duration 20 -cmt duration=20_exp_w=50
# single8 exp average
python train.py -m throughput -roadnet single8 -green_duration 20 -cmt duration=20_exp

# single4 exp average
python train.py -m queue -roadnet single4 -green_duration 20 -cmt duration=20_queue_500
python test.py -m queue -roadnet single4 -green_duration 20 -mln

python train.py -m throughput -roadnet single4 -green_duration 20 -cmt duration=20_exp
python test.py -m throughput -roadnet single4 -green_duration 20 -mln 

# uniform
python test.py -tsc uniform -roadnet single4 -green_duration 20

# sotl
python run_sotl.py -tsc sotl -roadnet single4 


#boxplot-delay
python delay.py -tsc uniform -roadnet single4 -green_duration 20
python delay.py -tsc dqn -m queue -roadnet single4 -green_duration 20 -mln 1
python delay.py -tsc dqn -m throughput -roadnet single4 -green_duration 20 -mln 7


#duration
python test.py -m throughput -roadnet single4 -green_duration 15 -mln