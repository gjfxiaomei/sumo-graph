import numpy as np
import os
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps
    
    def generate_specific(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated
        with open("roadnet/single.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                if step == math.ceil(np.mean(car_gen_steps)):
                    print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
            print("</routes>", file=routes)

    def generate_uniform(self, seed):
        """
        Generation of the route of every car for one episode
        """
        # np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("roadnet/single.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 5)  # choose a random source & destination
                    if route_straight == 1:
                        print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                else:  # car that turn -25% of the time the car turns
                    route_turn = np.random.randint(1, 9)  # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
            print("</routes>", file=routes)
    
    def generate_imbalance(self, seed):

        # np.random.seed(1)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("roadnet/imbalance.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_E" edges="W2TL TL2E"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/> 
            <route id="N_S" edges="N2TL TL2S"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                ns_or_we = np.random.uniform()
                if ns_or_we < 0.85:  # choose direction: straight or turn - 75% of times the car goes straight
                    up_or_down = np.random.randint(1,3)
                    if up_or_down == 1:
                        print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                else:  # car that turn -25% of the time the car turns
                    left_or_right = np.random.randint(1,3)
                    if left_or_right == 1:
                        print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
            print("</routes>", file=routes)


    def generate_biased(self, seed, bias):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("roadnet/single.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>

            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            # p = np.random.uniform()

            for car_counter, step in enumerate(car_gen_steps):
                u1 = np.random.uniform()
                if u1 < 0.5 + bias: #
                    u2 = np.random.uniform()
                    if u2 < 0.75:
                        stright = np.random.randint(1,3)
                        if stright == 1:
                            # S2TL-TL2N
                            print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                        else:
                            # N2TL-TL2S
                            print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        turn = np.random.randint(1,5)
                        if turn == 1:
                            #S2TL-TL2W
                            print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif turn == 2:
                            #S2TL-TL2E
                            print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif turn == 3:
                            #N2TL-TL2W:
                            print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                        else:
                            #N2TL-TL2E
                            print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                else:
                    u2 = np.random.uniform()
                    if u2 < 0.75:
                        #go stright
                        stright = np.random.randint(1,3)
                        if stright == 1:
                            #E2TL-TL2W
                            print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                        else:
                            #W2TL-TL2E
                            print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        turn = np.random.randint(1,5)
                        if turn == 1:
                            #E2TL-TL2N
                            print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif turn == 2:
                            #E2TL-TL2S
                            print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif turn == 3:
                            #W2TL-TL2N:
                            print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)
                        else:
                            #W2TL-TL2S
                            print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="best" departSpeed="10" />' % (car_counter, step), file=routes)

            print("</routes>", file=routes)
