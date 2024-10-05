# sleep 10
# python my_train.py --setting candidate --seed 8
# sleep 30
# python my_train.py --setting candidate --seed 9
# sleep 30
# python my_train.py --setting blind --seed 0
# sleep 30
# python my_train.py --setting blind --seed 1

sleep 10
python my_show.py --setting candidate --seed 8
sleep 30
python my_show.py --setting candidate --seed 9
sleep 30
python my_show.py --setting blind --seed 0
sleep 30
python my_show.py --setting blind --seed 1