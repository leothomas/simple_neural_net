# SNAKES GAME
# Use ARROW KEYS to play, SPACE BAR for pausing/resuming and Esc Key for exiting

import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint, uniform
import numpy as np
from progress.bar import Bar
from matplotlib import pyplot as plt
from simple_neural_net import Network

class Snake:

    DIRECTIONS = [KEY_UP, KEY_RIGHT, KEY_DOWN, KEY_LEFT]
    HISTORY = {'inputs': [], 'outputs': []}

    def __init__(self, brain=None):

        # TODO: parametrize x and y dimensions of window
        self.__MAX_DIST = self.calculate_distance(
            [0, 0], [30, 30]
        )

        if brain:
            self.brain = brain
        else:
            self.brain = Network(shape=[6,18, 18, 3])
            for neuron in self.brain.output_layer:
                neuron.transfer= "linear"

        head = [
            # start 3 up to account for body
            randint(0, 29),
            randint(4, 29),
        ]
        self.body = [
            head, 
            [head[0], head[1]-1], 
            [head[0], head[1]-2],
            [head[0], head[1]-3]
        ]

        self.generate_new_food()

    @property
    def head(self):
        return self.body[0]

    def calculate_distance(self, point1, point2):
        # distance is normalized to max distance (center to corner)
        return np.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)

    def calculate_angle(self, origin, point):
        return np.arcsin((point[1]-origin[1])/self.calculate_distance(origin, point)) * (180/np.pi)

    def calculate_inputs(self, key):
        # inputs are: position of snake, position of food, and direction of snake

        inputs = [
            # distance from the wall along the x-axis
            self.calculate_distance(
                self.head, [self.head[0], 0])/self.__MAX_DIST,
            # distance from the wall along the y-axis
            self.calculate_distance(
                self.head, [0, self.head[1]])/self.__MAX_DIST
        ]

        # divide by 180 to normalize to [-1, 1]
        angle_to_food = self.calculate_angle(self.head, self.food)/180
        inputs.append(angle_to_food)

        distance_to_food = self.calculate_distance(
            self.head, self.food)/self.__MAX_DIST
        inputs.append(distance_to_food)

        body_x_distance = self.__MAX_DIST
        body_y_distance = self.__MAX_DIST

        for part in self.body:
            if self.head[0] == part[0]:
                body_x_distance = self.calculate_distance(
                    self.head, part)/self.__MAX_DIST
                break
            if self.head[1] == part[1]:
                body_y_distance = self.calculate_distance(
                    self.head, part)/self.__MAX_DIST
                break

        inputs.append(body_x_distance)  # snake "sees" itself along x
        inputs.append(body_y_distance)  # snake "sees" itself along y

        # inputs.extend([int(key == key_opt)
        #                for key_opt in self.DIRECTIONS])  # snake direction
        return inputs

    def out_of_bounds(self):

        return (
            self.head[0] <= 1
            or self.head[0] >= 29
            or self.head[1] <= 1
            or self.head[1] >= 29
        )

    def add_body_segment(self, key):
        self.body.insert(0, [
            self.head[0] + (key == KEY_LEFT and -1) + (key == KEY_RIGHT and 1),
            self.head[1] + (key == KEY_DOWN and -1) + (key == KEY_UP and 1),
        ])

    def generate_new_food(self):
        food = []
        while food == []:
            food = [
                randint(1, 29),
                randint(1, 29),
            ]
            if food in self.body:
                food = []

        self.food = food

    def interpret(self, prev_key):
        outputs = [neuron.output for neuron in self.brain.output_layer]
        max_output_index = outputs.index(max(outputs))

        if max_output_index == 1:  # go straight
            return prev_key

        if max_output_index == 0:  # turn left
            return self.DIRECTIONS[
                (self.DIRECTIONS.index(prev_key) - 1) % len(self.DIRECTIONS)
            ]

        if max_output_index == 2:  # turn right
            return self.DIRECTIONS[
                (self.DIRECTIONS.index(prev_key) + 1) % len(self.DIRECTIONS)
            ]

    def decide(self, prev_key):

        inputs = self.calculate_inputs(prev_key)

        self.HISTORY['inputs'].append(inputs)

        self.brain.forward_pass(inputs)

        choice = self.interpret(prev_key)

        self.HISTORY['outputs'].append(choice)

        return choice


def setup_game(snake=None, visual=False):
    
    if not snake:
        snake = Snake()
    
    if visual:
        curses.initscr()

        # newwin(nlines, ncolumns)
        # this means that snake[:][0] is the y value, and snake[:][1] is the x value
        win = curses.newwin(31, 31, 0, 0)
        win.keypad(1)
        curses.noecho()
        curses.curs_set(0)
        win.border(0)
        win.nodelay(1)

        # Prints the food
        win.addch(snake.food[0], snake.food[1], '@')

        return win, snake
    
    return None, snake


def play_game(snake, win=None):

    # Initializing values
    score = 0

    lives = 200

    # TODO: make the inital location, direction and food location random.
    # It shouldn't start very close to a wall, and headed directly for it.
    key = KEY_LEFT
    if uniform(0,1)>0.5:
        key = KEY_RIGHT
    
    # While Esc key is not pressed
    while True:
        if win: 
            win.border(0)
            # Printing 'Score' and
            win.addstr(0, 2, 'Score : ' + str(score) + ' ')
            

            # Increases the speed of Snake as its length increases
            # win.timeout(150 - (len(snake)/5 + len(snake)/10) % 120)
            # block the screen and wait for user input
            win.timeout(15)

        # Previous key pressed
        prev_key = key

        if win: 
            event = win.getch()
            key = key if event == -1 else event

            # If SPACE BAR is pressed, wait for another
            if key == ord(' '):
                # one (Pause/Resume)

                key = -1
                while key != ord(' '):
                    key = win.getch()
                    if key == 27:
                        break

                key = prev_key
                continue

        if lives <= 0:
            break

        score += 1
        lives -= 1
        key = snake.decide(prev_key)

        # If an invalid key is pressed
        if key not in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, 27]:
            key = prev_key

        # Calculates the new coordinates of the head of the snake. NOTE: len(snake) increases.
        # This is taken care of later at [1].

        snake.add_body_segment(key)

        # Exit if snake crosses the boundaries (Uncomment to enable)
        if snake.out_of_bounds():
            break

        # If snake runs over itself
        if snake.head in snake.body[1:]:
            break

        # When snake eats the food
        if snake.head == snake.food:
            
            snake.generate_new_food()

            # max 500 lives
            if lives <= 400:
                lives += 100
            
            if win:  
                try:
                    win.addch(snake.food[0], snake.food[1], '@')
                except Exception as e:
                    print ("Attempted to add food at: ", snake.food)
                    break
                    
                
                
        else:
            # [1] If it does not eat the food, length decreases
            last = snake.body.pop()
            if win: 
                try:
                    win.addch(last[0], last[1], ' ')
                except Exception as e:
                    
                    print ("Attempted to remvove tail from: ", last)
                    break
        if win: 
            # draw a characters at the new head corrdinates to make the
            # snake "advance" by one spot
            win.addch(snake.head[0], snake.head[1], '#')
            
        if lives < 1:
            break

    if win: 
        curses.endwin()

    return score, snake


def generation(snakes=None, visual=False):

    generation = []

    for i in range(1000):

        if snakes:
            snake = snakes[i]
            win, snake = setup_game(snake=snake, visual=visual)

        else:
            win, snake = setup_game(visual=visual)

        final_score, snake = play_game(snake=snake, win=win)
    
        generation.append({
            'score': final_score,
            'snake': snake
        })
    select_percent = 0.1
    num_snakes = int(len(generation)*select_percent)
    top_snakes = sorted(generation, key=lambda x: x['score'], reverse=True)[:num_snakes]
    top_snakes_score = sum([x['score'] for x in top_snakes])/len(top_snakes)
    return top_snakes, top_snakes_score

# TODO: revisit this, use numpy function if possible (currently
# the issue is that the biases/weights are not regularly shaped)
def flatten(l):
    if l == []:
        return l
    if isinstance(l[0], list):
        return flatten(l[0]) + flatten(l[1:])
    return l[:1] + flatten(l[1:])


def reproduce(snake1, snake2):
    b1 = snake1['snake'].brain.biases
    b2 = snake2['snake'].brain.biases

    w1 = snake1['snake'].brain.weights
    w2 = snake2['snake'].brain.weights

    # generate new snake by randomly selecting biases
    # and weights from each parent with 50/50 proability

    b1_flat = flatten(b1)

    w1_flat = flatten(w1)

    b1_initial = len(b1_flat)
    w1_initial = len(w1_flat)

    crossover_pt = uniform(0, 1)

    for li, layer in enumerate(b1):
        for ni, neuron in enumerate(b1[li]):

            #if len(b1_flat) > crossover_pt * b1_initial:
            if uniform(0,1) <= 0.5:
                temp = b2[li][ni]
                b2[li][ni] = b1[li][ni]
                b1[li][ni] = temp

            # random mutation                
            if uniform(0, 1) <= 0.1:
                b2[li][ni] = mutation(b2[li][ni])
                b1[li][ni] = mutation(b1[li][ni])
               
            for si, synapses in enumerate(w1[li][ni]):
                #if len(w1_flat) > crossover_pt * w1_initial:
                if uniform(0,1) <= 0.5:
                    temp = w2[li][ni][si]
                    w2[li][ni][si] = w1[li][ni][si]
                    w1[li][ni][si] = temp

                #random mutation
                if uniform(0, 1) <= 0.1:
                    w2[li][ni][si] = mutation(w2[li][ni][si])
                    w1[li][ni][si] = mutation(w1[li][ni][si])

    brain1 = Network(shape=[6, 18, 18, 3])
    brain2 = Network(shape=[6, 18, 18, 3])
    
    for neuron in brain1.output_layer:
        neuron.transfer = 'linear'
    
    for neuron in brain2.output_layer:
        neuron.transfer = 'linear'
    
    brain1.biases = b1
    brain1.weights = w1
    
    brain2.biases = b2
    brain2.weights = w2
    
    return [Snake(brain=brain1), Snake(brain=brain2)]

# add small, random mutation (without exceeding [1,1])
def mutation(value):
    mut = uniform(-1, 1)
    while value + mut < -1 or value + mut > 1:
        mut = uniform(-1, 1)
    return  value + mut

def roulette_select(snakes, pick):
    current = 0
    for snake in snakes:
        current += snake['score']
        if current > pick:
            return snake


def get_pair(snakes):

    total_parents_score = sum([snake['score'] for snake in snakes])

    return (
        roulette_select(snakes, uniform(0, total_parents_score)), 
        roulette_select(snakes, uniform(0, total_parents_score))
    )


def crossover(snakes):

    # perserve all parents from the previous generation, in 
    # case none of the snakes outperform the parents
    new_snakes = [Snake(brain=snake['snake'].brain) for snake in snakes]

    while len(new_snakes) < 1000:
        snake1, snake2 = get_pair(snakes)

        new_snakes.extend(reproduce(snake1, snake2))
        
    return new_snakes


if __name__ == "__main__":

    top_snakes, top_snakes_score = generation(visual=False)
    new_snakes = crossover(top_snakes)
    
    gen_count = 0   
    scores = []
    max_gen = 30   
    bar = Bar('Training', max=max_gen)
    while gen_count <= max_gen: 
       
        top_snakes, top_snakes_score = generation(new_snakes, visual=False)
        
        scores.append(top_snakes_score)
        
        new_snakes = crossover(top_snakes)
        gen_count += 1

        bar.next()
    bar.finish()

    print ("Snakes after 200 generations; ")
    print (top_snakes)
    plt.plot(scores)
    plt.show()
    
    play_visual = input("Go through one generation visually?\n")
    if play_visual.lower() in ['yes', 'y', 'ye']:
        generation(snakes=new_snakes, visual=True)

    # print("Final score: ", final_score)

    # print("Decision Inputs: ")
    # for inp in snake.HISTORY['inputs']:
    #    print(inp)
    # print("Decisions: ", snake.HISTORY['outputs'])
