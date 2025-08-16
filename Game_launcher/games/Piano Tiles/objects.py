import pygame
import random

# 동적 크기 계산을 위한 함수
def get_screen_size():
    info = pygame.display.Info()
    width = info.current_w
    height = info.current_h
    return int(width * 1.0), int(height * 1.0)

def get_scale_factors():
    WIDTH, HEIGHT = get_screen_size()
    ORIGINAL_WIDTH, ORIGINAL_HEIGHT = 432, 768
    return WIDTH / ORIGINAL_WIDTH, HEIGHT / ORIGINAL_HEIGHT

# 기본값 (게임 시작 시 업데이트됨)
SCREEN = WIDTH, HEIGHT = 432, 768
TILE_WIDTH = WIDTH // 4
TILE_HEIGHT = 195

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (30, 144, 255)
BLUE2 = (2, 239, 239)
PURPLE = (191, 64, 191)

# 타일 ID 카운터 (전역 변수)
tile_id_counter = 0

def reset_tile_id_counter():
	global tile_id_counter
	tile_id_counter = 0

class Tile(pygame.sprite.Sprite):
	def __init__(self, x, y, win, game_x=0, game_y=0):
		super(Tile, self).__init__()
		
		global tile_id_counter
		tile_id_counter += 1
		self.tile_id = tile_id_counter  # 고유 ID 할당

		self.win = win
		self.game_x = game_x
		self.game_y = game_y
		self.x, self.y = x, y
		self.color = BLACK
		self.alive = True
		self.alpha = 255  # 투명도 추가

		# 동적 크기 계산 (게임 창 크기에 맞춰 확대)
		_, HEIGHT = get_screen_size()
		scale_factor = HEIGHT / 768
		tile_width = int((432 // 4) * scale_factor)
		tile_height = int(195 * scale_factor)

		self.surface = pygame.Surface((tile_width, tile_height), pygame.SRCALPHA)
		self.rect = self.surface.get_rect()
		self.rect.x = x
		self.rect.y = y

		self.center = tile_width//2, tile_height//2 + int(15 * scale_factor)
		self.line_start = self.center[0], self.center[1]-int(18 * scale_factor)
		self.line_end = self.center[0], int(20 * scale_factor)

	def update(self, speed):
		self.rect.y += speed
		# 게임 창 높이로 제한
		_, HEIGHT = get_screen_size()
		if self.rect.y >= HEIGHT:
			self.kill()

		if self.alive:
			# 동적 크기 계산
			_, HEIGHT = get_screen_size()
			scale_factor = HEIGHT / 768
			tile_width = int((432 // 4) * scale_factor)
			tile_height = int(195 * scale_factor)
			
			# 투명도 적용
			color_with_alpha = (*self.color, self.alpha)
			purple_with_alpha = (*PURPLE, self.alpha)
			blue2_with_alpha = (*BLUE2, self.alpha)
			blue_with_alpha = (*BLUE, self.alpha)
			
			pygame.draw.rect(self.surface, color_with_alpha, (0,0, tile_width, tile_height))
			pygame.draw.rect(self.surface, purple_with_alpha, (0,0, tile_width, tile_height), int(4 * scale_factor))
			pygame.draw.rect(self.surface, blue2_with_alpha, (0,0, tile_width, tile_height), int(2 * scale_factor))
			pygame.draw.line(self.surface, blue_with_alpha, self.line_start, self.line_end, int(3 * scale_factor))
			pygame.draw.circle(self.surface, blue_with_alpha, self.center, int(15 * scale_factor), int(3 * scale_factor))
		else:
			# 동적 크기 계산
			_, HEIGHT = get_screen_size()
			scale_factor = HEIGHT / 768
			tile_width = int((432 // 4) * scale_factor)
			tile_height = int(195 * scale_factor)
			pygame.draw.rect(self.surface, (0,0,0, 90), (0,0, tile_width, tile_height))
			
		# 게임 창 영역에만 그리기
		self.win.blit(self.surface, (self.game_x + self.rect.x, self.game_y + self.rect.y))

class Text(pygame.sprite.Sprite):
	def __init__(self, text, font, pos, win, game_x=0, game_y=0):
		super(Text, self).__init__()
		self.win = win
		self.game_x = game_x
		self.game_y = game_y

		self.x,self.y = pos
		self.initial = self.y
		self.image = font.render(text, True, (255, 255, 255))

	def update(self, speed):
		self.y += speed
		if self.y - self.initial >= 100:
			self.kill()

		# 게임 창 영역에만 그리기
		self.win.blit(self.image, (self.game_x + self.x, self.game_y + self.y))

class Counter(pygame.sprite.Sprite):
	def __init__(self, win, font, game_x=0, game_y=0):
		super(Counter, self).__init__()

		self.win = win
		self.game_x = game_x
		self.game_y = game_y
		self.font = font
		self.index = 1
		self.count = 3

	def update(self):
		if self.index % 30 == 0:
			self.count -= 1

		self.index += 1

		if self.count > 0:
			self.image = self.font.render(f'{self.count}', True, (255, 255, 255))
			WIDTH, HEIGHT = get_screen_size()
			# 게임 창 영역에만 그리기
			self.win.blit(self.image, (self.game_x + WIDTH//2-16, self.game_y + HEIGHT//2-25))

class Square(pygame.sprite.Sprite):
	def __init__(self, win, game_x=0, game_y=0):
		super(Square, self).__init__()

		self.win = win
		self.game_x = game_x
		self.game_y = game_y
		self.color = (255, 255, 255)
		self.speed = 3
		self.angle = 0

		# 동적 크기 계산 (게임 창 크기에 맞춰 확대)
		WIDTH, HEIGHT = get_screen_size()
		scale_factor = HEIGHT / 768
		self.side = random.randint(int(15 * scale_factor), int(40 * scale_factor))
		# 게임 창 내에서만 생성되도록 x좌표 계산
		GAME_WIDTH = int(432 * (HEIGHT / 768))  # 게임 창의 실제 너비
		x = random.randint(self.side, GAME_WIDTH - self.side)
		y = 0  # 게임 창 제일 위에서 시작

		self.surface = pygame.Surface((self.side, self.side), pygame.SRCALPHA)
		# center는 게임 창 내에서만 맞추기
		self.rect = self.surface.get_rect()
		self.rect.x = x
		self.rect.y = y

	def update(self):
		center = self.rect.center
		self.angle = (self.angle + self.speed) % 360
		image = pygame.transform.rotate(self.surface , self.angle)
		self.rect = image.get_rect()
		self.rect.center = center

		self.rect.y += 1.5

		# 게임 창 높이로 제한
		_, HEIGHT = get_screen_size()
		if self.rect.y >= HEIGHT:
			self.kill()

		pygame.draw.rect(self.surface, self.color, (0,0, self.side, self.side), 4)
		pygame.draw.rect(self.surface, (30, 144, 255, 128), (2,2, self.side-4, self.side-4), 2)
		# 게임 창 영역에만 그리기
		self.win.blit(image, (self.game_x + self.rect.x, self.game_y + self.rect.y))

class Button(pygame.sprite.Sprite):
	def __init__(self, img, scale, x, y):
		super(Button, self).__init__()
		
		self.scale = scale
		self.image = pygame.transform.scale(img, self.scale)
		self.rect = self.image.get_rect()
		self.rect.x = x
		self.rect.y = y

		self.clicked = False

	def update_image(self, img):
		self.image = pygame.transform.scale(img, self.scale)

	def draw(self, win):
		action = False
		pos = pygame.mouse.get_pos()
		if self.rect.collidepoint(pos):
			if pygame.mouse.get_pressed()[0] and not self.clicked:
				action = True
				self.clicked = True

			if not pygame.mouse.get_pressed()[0]:
				self.clicked = False

		win.blit(self.image, self.rect)
		return action