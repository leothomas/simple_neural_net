# SNAKES GAME
# Use ARROW KEYS to play, SPACE BAR for pausing/resuming and Esc Key for exiting

import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
import numpy as np
from progress.bar import Bar
from matplotlib import pyplot as plt
# from simple_neural_net import Network
from matrix_neural_net import Network


class Game:
    XMAX = 10 
    YMAX = 10
    def __init__(self, snake, visual=False):
        
        self.snake = snake 
        # TODO: make these next two lines cleaner
        self.food = None
        self.generate_new_food()
        self.win = None

        if visual:
            curses.initscr()

            # newwin(nlines, ncolumns)
            # this means that snake[:][0] is the y value, and snake[:][1] is the x value
            self.win = curses.newwin(self.XMAX+1, self.YMAX+1, 0, 0)
            self.win.keypad(1)
            curses.noecho()
            curses.curs_set(0)
            self.win.border(0)
            self.win.nodelay(1)

            # Prints the food
            self.win.addch(self.food[0], self.food[1], '@')

            self.win
    
    def snake_out_of_bounds(self):

        return (
            self.snake.head[0] <= 0
            or self.snake.head[0] >= self.XMAX
            or self.snake.head[1] <= 0
            or self.snake.head[1] >= self.YMAX
        )

    def generate_new_food(self):
        food = []
        while food == []:
            food = [
                np.random.randint(1, self.XMAX-1),
                np.random.randint(1, self.YMAX-1),
            ]
            if food in self.snake.body:
                food = []

        self.food = food    

    def play(self):

        # Initializing values
        score = 0
        steps = 0
        food_consumed = 0
        lives = 100

        key = np.random.choice(self.snake.DIRECTIONS)
        
        # While Esc key is not pressed
        while True:
            if self.win: 
                self.win.border(0)
                # Printing 'Score' and
                self.win.addstr(0, 2, 'Score : ' + str(food_consumed) + ' ')
                
                # Increases the speed of Snake as its length increases
                # self.win.timeout(150 - (len(snake)/5 + len(snake)/10) % 120)
                # block the screen and wait for user input
                self.win.timeout(100)

            # Previous key pressed
            prev_key = key

            if self.win: 
                event = self.win.getch()
                key = key if event == -1 else event

                # If SPACE BAR is pressed, wait for another
                if key == ord(' '):
                    # one (Pause/Resume)

                    key = -1
                    while key != ord(' '):
                        key = self.win.getch()
                        if key == 27:
                            break

                    key = prev_key
                    continue

            if lives <= 0:
                break
            
            steps +=1
            #score += 1 # each new move adds a point
            lives -= 1
            key = self.snake.decide(prev_key, self.food)

            # If an invalid key is pressed
            if key not in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, 27]:
                key = prev_key

            # Calculates the new coordinates of the head of the snake. NOTE: len(snake) increases.
            # This is taken care of later at [1].

            self.snake.add_body_segment(key)

            # Exit if snake crosses the boundaries (Uncomment to enable)
            if self.snake_out_of_bounds():
                break

            # If snake runs over itself
            if self.snake.head in self.snake.body[1:]:
                break

            # When self.snake eats the food
            if self.snake.head == self.food:
                
                # eating food adds 10 points, 
                # I want to prioritize food seeking over simply surviving
                # (which can be acheived by simply going in circles)
                #score += 100
                food_consumed += 1
                self.generate_new_food()

                lives += 100
                if lives >500:
                    lives = 500
                
                if self.win:  
                    self.win.addch(self.food[0], self.food[1], '@')
                    
            else:
                # [1] If it does not eat the food, length decreases
                last = self.snake.body.pop()
                if self.win: 
                   self.win.addch(last[0], last[1], ' ')
                
            if self.win: 
                # draw a characters at the new head corrdinates to make the
                # snake "advance" by one spot
                self.win.addch(self.snake.head[0], self.snake.head[1], '#')
                
            if lives < 1:
                break

        if self.win: 
            curses.endwin()

        #score = steps + ((2**food_consumed) + (food_consumed**2.1 * 500)) - ((food_consumed**1.2)*(0.25*steps)**1.3)
        #score = steps + (food_consumed * 10)
        score = steps

        return score, self.snake


class Snake:

    DIRECTIONS = [KEY_UP, KEY_RIGHT, KEY_DOWN, KEY_LEFT]
    HISTORY = {'inputs': [], 'outputs': []}
    XMAX = 10
    YMAX = 10
    def __init__(self, brain=None):

        # TODO: parametrize x and y dimensions of window
        self.__MAX_DIST = self.calculate_distance(
            [0, 0], [self.XMAX, self.YMAX]
        )
        
        if brain:
            self.brain = brain
        else:
            self.brain = Network(
                shape=[10, 8, 8, 3], 
                activation="tanh",
                output_activation="linear"
                )
            

        head = [
            # start 3 up to account for body
            np.random.randint(0, self.XMAX),
            np.random.randint(4, self.YMAX),
        ]
        self.body = [
            head, 
            [head[0], head[1]-1], 
            [head[0], head[1]-2],
            [head[0], head[1]-3]
        ]

    @property
    def head(self):
        return self.body[0]

    def calculate_distance(self, point1, point2):
        # distance is normalized to max distance (center to corner)
        return np.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)

    def calculate_angle(self, origin, point, degrees=False):
        epsilon = 10**-9 # add a small value to divisor to avoid division by zero error
        alpha = np.arctan((point[0]-origin[0])/(point[1]-origin[1] + epsilon)) 
        
        if degrees:
            return alpha * (180/np.pi)
        
        return  alpha

        
    def calculate_inputs(self, key, food):

        # input vector:
        # 1) left blocked (by wall or body): 0/1
        # 2) right  blocked (by wall or body): 0/1
        # 3) above  blocked (by wall or body): 0/1
        # 4) below blocked (by wall or body): 0/1
        # 5) delta_x to apple: Int
        # 6) delta_y to apple: Int
        # 7) snake moving left: 0/1
        # 8) snake moving right: 0/1
        # 9) snake moving up: 0/1
        # 10) snake moving down: 0/1

        inputs = np.zeros(10)
        
        # # distance to walls: 
        # vectors_from_head = [
        #     # define all 8 directional vectors from the snakes head
        #     # cross product of each vector with:
        #     # 1) food vector
        #     # 2) any of the body parts
        #     # to determin if they lie on the same "line" (
        #     # i.e. the snake "sees" that item: wall, food or self )
        # ]
        # # 0 deg
        # self.head[1]
        # # 45 deg
        # np.sqrt(self.head[0]**2 + self.head[1]**2)
        # # 90 deg
        # self.head[0]
        # # 135 deg
        # np.sqrt((self.XMAX-self.head[0])**2 + self.head[1])
        # # 180 deg         
        # self.XMAX-self.head[0]
        # # 225 deg
        # np.sqrt(
        #     (self.XMAX-self.head[0])**2 + 
        #     (self.YMAX-self.head[1])**2
        # )
        # # 270 deg 
        # self.YMAX-self.head[1]
        # # 315 deg
        # np.sqrt(
        #     self.head[0]**2 + 
        #     (self.YMAX-self.head[1])**2
        # )

        # # seeing food
        
        # wall to the left
        inputs[0] = int(self.head[0] == 1)
        
        # body part to the left
        for part in self.body:
            # snake body part directly to the left of the snake's head
            if (part[0], part[1]) == (self.head[0] - 1, self.head[1]):
                inputs[0] = 1
                break
        
        # wall to the right
        inputs[1] = int(self.head[0] == self.XMAX-1)
            
        # body part to the right
        for part in self.body:
            # snake body part directly to the right of the snake's head
            if (part[0], part[1]) == (self.head[0] + 1, self.head[1]):
                inputs[0] = 1
                break
        
        # wall above
        inputs[2] = int(self.head[1] == 1)
        
        # body part above
        for part in self.body:
            # snake body part directly to the left of the snake's head
            if (part[0], part[1]) == (self.head[0], self.head[1]-1):
                inputs[2] = 1
                break
        
        # wall below
        inputs[3] = int(self.head[1] == self.YMAX-1)
        
        # body part to the right
        for part in self.body:
            # snake body part directly to the right of the snake's head
            if (part[0], part[1]) == (self.head[0], self.head[1]+1):
                inputs[3] = 1
                break
        
        inputs[4] = (self.head[0] - food[0])#/self.__MAX_DIST # delta x to food
        inputs[5] = (self.head[1] - food[1])#/self.__MAX_DIST # delta y to food

        # snake direction vector
        inputs[6] = int(key == KEY_LEFT)
        inputs[7] = int(key == KEY_RIGHT)
        inputs[8] = int(key == KEY_UP)
        inputs[9] = int(key == KEY_DOWN)
        
        return inputs 

    def add_body_segment(self, key):
        self.body.insert(0, [
            self.head[0] + (key == KEY_LEFT and -1) + (key == KEY_RIGHT and 1),
            self.head[1] + (key == KEY_DOWN and -1) + (key == KEY_UP and 1),
        ])


    def interpret(self, prev_key, output):
    
        max_output_index = np.argmax(output)

        # return self.DIRECTIONS[max_output_index]

        if max_output_index == 1:  # go straight
            # print ("Go STRAIGHT!")
            return prev_key

        if max_output_index == 0:  # turn left
            # print ("Go LEFT!")
            return self.DIRECTIONS[
                (self.DIRECTIONS.index(prev_key) - 1) % len(self.DIRECTIONS)
            ]

        if max_output_index == 2:  # turn right
            # print ("GO RIGHT!")
            return self.DIRECTIONS[
                (self.DIRECTIONS.index(prev_key) + 1) % len(self.DIRECTIONS)
            ]

    def decide(self, prev_key, food):

        inputs = self.calculate_inputs(prev_key, food)

        output = self.brain.forward_pass(inputs)

        choice = self.interpret(prev_key, output)

        return choice



def run_generation(num_snakes, snakes=None, visual=False):

    generation = []

    for i in range(num_snakes):

        if snakes:
            snake = snakes[i]
            game = Game(snake=snake, visual=visual)

        else:
            game = Game(snake = Snake(), visual=visual)

        score, snake = game.play()
    
        generation.append({
            'score': score, 
            'snake': snake
        })

    select_percent = 0.25 # select top 25% of snakes
    num_top_snakes = int(len(generation)*select_percent)
    
    top_snakes = sorted(generation, key=lambda x: x['score'], reverse=True)[:num_top_snakes]
    
    # average score for the top snakes
    top_snakes_score = sum([x['score'] for x in top_snakes])/len(top_snakes)
    top_snakes_score = top_snakes[0]['score']
    return top_snakes, top_snakes_score


def reproduce(parent1, parent2):
    b1 = parent1['snake'].brain.biases
    b2 = parent2['snake'].brain.biases

    w1 = parent1['snake'].brain.weights
    w2 = parent2['snake'].brain.weights
    

    # TODO: figure out how to select the layers randomly from one or the other
    # without having to loop through them 
    # w1_indices = np.random.choice(np.arange(len(w1)), crossover_point)

    for li, _ in enumerate(w1):
        # store original weight matrix shape
        shape = w1[li].shape
        weight_layer_1 = w1[li].flatten()
        weight_layer_2 = w2[li].flatten()
        
        # iterate through weight layer switching and mutating weights
        weight_layer_1, weight_layer_2 = crossover(weight_layer_1, weight_layer_2)
        weight_layer_1, weight_layer_2 = mutate(weight_layer_1), mutate(weight_layer_2)

        # reformat to original shape
        w1[li] = weight_layer_1.reshape(shape)
        w2[li] = weight_layer_2.reshape(shape)

        bias_layer_1 = b1[li]
        bias_layer_2 = b2[li]

        # iterate through biases switching and mutating 
        bias_layer_1, bias_layer_2 = crossover(bias_layer_1, bias_layer_2)
        bias_layer_1, bias_layer_2  = mutate(bias_layer_1), mutate(bias_layer_2)
        
        b1[li] = bias_layer_1
        b2[li] = bias_layer_2
        
    brain1 = Network(
        shape=[10, 8, 8, 3],
        activation='tanh',
        output_activation='linear'
    )
    brain1.biases = b1
    brain1.weights = w1
    
    brain2 = Network(
        shape=[10, 8, 8, 3],
        activation='tanh',
        output_activation='linear'
        )
    brain2.biases = b2
    brain2.weights = w2
    
    return [Snake(brain=brain1), Snake(brain=brain2)]

def mutate(values):
    
    # mutate a small amount of values
    mutation_prob = 0.1

    # randomly select the above percent of indices to be mutated
    indices_to_mutate = np.random.choice(
        np.arange(len(values)),  
        size=int(mutation_prob * len(values)), 
        replace=False
    )

    for mi in indices_to_mutate:
        values[mi] = np.random.uniform(-1,1)

    return values

# TODO: implement 2point crossover, which has been proven more efficient
# than single point
def crossover(values1, values2):
    #cpoint = np.random.choice(np.arange(len(values1)))
    cpoint = int(0.5*len(values1))
    r1 = np.append(values1[:cpoint], values2[cpoint:])
    r2 = np.append(values2[:cpoint], values1[cpoint:])
    return r1, r2
    
def roulette_select(snakes, pick):
    current = 0
    for snake in snakes:
        current += snake['score']
        if current > pick:
            return snake


def get_pair(snakes):

    total_parents_score = sum([snake['score'] for snake in snakes])

    return (
        roulette_select(snakes, np.random.uniform(0, total_parents_score)), 
        roulette_select(snakes, np.random.uniform(0, total_parents_score))
    )


def get_new_generation(parents, num_snakes):

    # perserve all parents from the previous generation, in 
    # case none of the child snakes outperform the parents
    new_snakes = [Snake(brain=parent['snake'].brain) for parent in parents]

    while len(new_snakes) < num_snakes:
        
        parent1, parent2 = get_pair(parents)

        new_snakes.extend(reproduce(parent1, parent2))
        
    return new_snakes

class BarWithScore(Bar):
    suffix = 'Generation %(index)d/%(max)d - Average Score: %(avg_score).1f'
    @property
    def avg_score(self):
        return self.score
    @avg_score.setter
    def avg_score(self, score):
        self.score = score


if __name__ == "__main__":

    print ("Starting...")
    snakes_per_gen = 100    
    top_snakes, top_snakes_score = run_generation(visual=False, num_snakes=snakes_per_gen)
    
    print ("Crossing over")
    new_snakes = get_new_generation(top_snakes, num_snakes=snakes_per_gen)
    
    print ("Done")

    gen_count = 0   
    scores = []
    max_gen = 150

    bar = BarWithScore('Training ', max=max_gen)
    bar.score = top_snakes_score
    while gen_count <= max_gen:  
       
        top_snakes, top_snakes_score = run_generation(num_snakes=snakes_per_gen, snakes=new_snakes, visual=False)
        
        scores.append(top_snakes_score)
    
        new_snakes = get_new_generation(top_snakes, num_snakes=snakes_per_gen)
        gen_count += 1
        bar.score = top_snakes_score
        bar.next()
    bar.finish()

    print ("Snakes after %i generations: "%max_gen) 
    print (top_snakes)
    plt.plot(scores)
    plt.show()
    
    play_visual = input("Go through one generation visually?\n")
    if play_visual.lower() in ['yes', 'y', 'ye']:
        run_generation(num_snakes=snakes_per_gen, snakes=new_snakes, visual=True)

    # print("Final score: ", final_score)

    # print("Decision Inputs: ")
    # for inp in snake.HISTORY['inputs']:
    #    print(inp)
    # print("Decisions: ", snake.HISTORY['outputs'])
