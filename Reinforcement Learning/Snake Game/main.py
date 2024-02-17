import pygame,sys,random
from pygame.math import Vector2
import numpy as np 

pygame.mixer.pre_init(44100,-16,2,512)
pygame.init()
# apple = pygame.image.load('Graphics/apple.png').convert_alpha()
game_font = pygame.font.Font('Font/PoetsenOne-Regular.ttf', 25)

class SNAKE:
	def __init__(self):
		self.body = [Vector2(5,10),Vector2(4,10),Vector2(3,10)]
		self.direction = Vector2(0,0)
		self.new_block = False

		self.head_up = pygame.image.load('Graphics/head_up.png').convert_alpha()
		self.head_down = pygame.image.load('Graphics/head_down.png').convert_alpha()
		self.head_right = pygame.image.load('Graphics/head_right.png').convert_alpha()
		self.head_left = pygame.image.load('Graphics/head_left.png').convert_alpha()
		
		self.tail_up = pygame.image.load('Graphics/tail_up.png').convert_alpha()
		self.tail_down = pygame.image.load('Graphics/tail_down.png').convert_alpha()
		self.tail_right = pygame.image.load('Graphics/tail_right.png').convert_alpha()
		self.tail_left = pygame.image.load('Graphics/tail_left.png').convert_alpha()

		self.body_vertical = pygame.image.load('Graphics/body_vertical.png').convert_alpha()
		self.body_horizontal = pygame.image.load('Graphics/body_horizontal.png').convert_alpha()

		self.body_tr = pygame.image.load('Graphics/body_tr.png').convert_alpha()
		self.body_tl = pygame.image.load('Graphics/body_tl.png').convert_alpha()
		self.body_br = pygame.image.load('Graphics/body_br.png').convert_alpha()
		self.body_bl = pygame.image.load('Graphics/body_bl.png').convert_alpha()
		self.crunch_sound = pygame.mixer.Sound('Sound/crunch.wav')
		self.screen = pygame.display.get_surface()

	def draw_snake(self, cell_size): 
		self.update_head_graphics()
		self.update_tail_graphics()

		for index,block in enumerate(self.body):
			x_pos = int(block.x * cell_size)
			y_pos = int(block.y * cell_size)
			block_rect = pygame.Rect(x_pos,y_pos,cell_size,cell_size)

			if index == 0:
				self.screen.blit(self.head,block_rect)
			elif index == len(self.body) - 1:
				self.screen.blit(self.tail,block_rect)
			else:
				previous_block = self.body[index + 1] - block
				next_block = self.body[index - 1] - block
				if previous_block.x == next_block.x:
					self.screen.blit(self.body_vertical,block_rect)
				elif previous_block.y == next_block.y:
					self.screen.blit(self.body_horizontal,block_rect)
				else:
					if previous_block.x == -1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == -1:
						self.screen.blit(self.body_tl,block_rect)
					elif previous_block.x == -1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == -1:
						self.screen.blit(self.body_bl,block_rect)
					elif previous_block.x == 1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == 1:
						self.screen.blit(self.body_tr,block_rect)
					elif previous_block.x == 1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == 1:
						self.screen.blit(self.body_br,block_rect)

	def update_head_graphics(self):
		head_relation = self.body[1] - self.body[0]
		if head_relation == Vector2(1,0): self.head = self.head_left
		elif head_relation == Vector2(-1,0): self.head = self.head_right
		elif head_relation == Vector2(0,1): self.head = self.head_up
		elif head_relation == Vector2(0,-1): self.head = self.head_down

	def update_tail_graphics(self):
		tail_relation = self.body[-2] - self.body[-1]
		if tail_relation == Vector2(1,0): self.tail = self.tail_left
		elif tail_relation == Vector2(-1,0): self.tail = self.tail_right
		elif tail_relation == Vector2(0,1): self.tail = self.tail_up
		elif tail_relation == Vector2(0,-1): self.tail = self.tail_down

	def move_snake(self):
		if self.new_block == True:
			body_copy = self.body[:]
			body_copy.insert(0,body_copy[0] + self.direction)
			self.body = body_copy[:]
			self.new_block = False
		else:
			body_copy = self.body[:-1]
			body_copy.insert(0,body_copy[0] + self.direction)
			self.body = body_copy[:]

	def add_block(self):
		self.new_block = True

	def play_crunch_sound(self):
		self.crunch_sound.play()

	def reset(self):
		self.__init__()
		# self.body = [Vector2(5,10),Vector2(4,10),Vector2(3,10)]
		# self.direction = Vector2(0,0)

class FRUIT:
	def __init__(self, cell_number):
		self.cell_number = cell_number
		self.randomize() 
		self.screen = pygame.display.get_surface() 
		self.apple = pygame.image.load('Graphics/apple.png').convert_alpha()

	def draw_fruit(self, cell_size):
		fruit_rect = pygame.Rect(int(self.pos.x * cell_size),int(self.pos.y * cell_size),cell_size,cell_size)
		self.screen.blit(self.apple,fruit_rect)
		#pygame.draw.rect(screen,(126,166,114),fruit_rect)

	def randomize(self):
		self.x = random.randint(0,self.cell_number - 1)
		self.y = random.randint(0,self.cell_number - 1)
		self.pos = Vector2(self.x,self.y)

class MAIN:
	def __init__(self):
		self.cell_size = 35
		self.cell_number = 15 
		self.screen = pygame.display.set_mode((self.cell_number * self.cell_size, self.cell_number * self.cell_size))
		self.snake = SNAKE()
		self.fruit = FRUIT(self.cell_number)  
		self.clock = pygame.time.Clock()
		self.gameOver = False 
		self.reward = 0 

	def update(self):
		self.snake.move_snake()
		self.check_collision()
		self.check_fail()

	def draw_elements(self):
		self.draw_grass()
		self.fruit.draw_fruit(self.cell_size)
		self.snake.draw_snake(self.cell_size) 
		self.draw_score()

	def check_collision(self):
		if self.fruit.pos == self.snake.body[0]:
			self.fruit.randomize()
			self.snake.add_block()
			self.snake.play_crunch_sound()
			self.reward = 10 

		for block in self.snake.body[1:]:
			if block == self.fruit.pos:
				self.fruit.randomize()

	def checkFailAgent(self, position):
		if position.x > self.cell_number or position.x < 0 or \
			position.y > self.cell_number or position.y < 0:
			return True 
		if position in self.snake.body[1:]:
			return True
		else:
			return False 

	def check_fail(self):
		if not 0 <= self.snake.body[0].x < self.cell_number or not 0 <= self.snake.body[0].y < self.cell_number:
			self.reward = -60
			self.gameOver = True

		for block in self.snake.body[1:]:
			if block == self.snake.body[0]:
				self.reward = -10
				self.gameOver = True 
		
	def game_over(self):
		self.__init__()
		# self.snake.reset()

	def draw_grass(self):
		grass_color = (167,209,61)
		for row in range(self.cell_number):
			if row % 2 == 0: 
				for col in range(self.cell_number):
					if col % 2 == 0:
						grass_rect = pygame.Rect(col * self.cell_size,row * self.cell_size,self.cell_size,self.cell_size)
						pygame.draw.rect(self.screen,grass_color,grass_rect)
			else:
				for col in range(self.cell_number):
					if col % 2 != 0:
						grass_rect = pygame.Rect(col * self.cell_size,row * self.cell_size,self.cell_size,self.cell_size)
						pygame.draw.rect(self.screen,grass_color,grass_rect)

	def movements(self, move):
		if move.argmax().item() == 0:
			if self.snake.direction.y != 1:
				self.snake.direction = Vector2(0, -1)
		elif move.argmax().item() == 1:
			if self.snake.direction.x != -1:
				self.snake.direction = Vector2(1, 0)
		elif move.argmax().item() == 2:
			if self.snake.direction.y != -1:
				self.snake.direction = Vector2(0, 1)
		else:
			if self.snake.direction.x != 1:
				self.snake.direction = Vector2(-1, 0)

	def draw_score(self):
		score_text = str(len(self.snake.body) - 3)
		score_surface = game_font.render(score_text,True,(56,74,12))
		score_x = int(self.cell_size * self.cell_number - 60)
		score_y = int(self.cell_size * self.cell_number - 40)
		score_rect = score_surface.get_rect(center = (score_x,score_y))
		apple_rect = self.fruit.apple.get_rect(midright = (score_rect.left,score_rect.centery))
		bg_rect = pygame.Rect(apple_rect.left,apple_rect.top,apple_rect.width + score_rect.width + 6,apple_rect.height)

		pygame.draw.rect(self.screen,(167,209,61),bg_rect)
		self.screen.blit(score_surface,score_rect)
		self.screen.blit(self.fruit.apple,apple_rect)
		pygame.draw.rect(self.screen,(56,74,12),bg_rect,2)

	def run(self, move):
		self.clock.tick(20)  
		
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()

		self.movements(move) 
		self.update()

		self.screen.fill((175, 215, 70))
		self.draw_elements()
		pygame.display.update()
		
		return self.reward, self.gameOver, len(self.snake.body) - 3 
	
# if __name__ == '__main__':
# 	game = MAIN() 
# 	while True:
# 		game.run(np.array([0, 0, 0, 1])) 