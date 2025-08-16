# Piano Tiles

# Author : Prajjwal Pathak (pyguru)
# Date : Thursday, 30 November, 2021

import json
import random
import pygame
from threading import Thread

from objects import Tile, Square, Text, Button, Counter, reset_tile_id_counter

pygame.init()
# 게임 창 크기를 흰색 창의 세로와 동일하게 설정
info = pygame.display.Info()
width = info.current_w
height = info.current_h

# 게임 창 크기 계산 (세로는 흰색 창과 동일, 가로는 같은 비율로 확대)
GAME_HEIGHT = height
GAME_WIDTH = int(432 * (height / 768))  # 원래 비율 유지하면서 확대

SCREEN = WIDTH, HEIGHT = GAME_WIDTH, GAME_HEIGHT
TILE_WIDTH = WIDTH // 4
TILE_HEIGHT = int(195 * (height / 768))  # 타일 높이도 같은 비율로 확대

# 전체 화면으로 설정
win = pygame.display.set_mode((width, height), pygame.NOFRAME)

# COLORS *********************************************************************

WHITE = (255, 255, 255)
GRAY = (75, 75, 75)
BLUE = (30, 144, 255)

# IMAGES *********************************************************************

bg_img = pygame.image.load('Assets/bg.png')
bg_img = pygame.transform.scale(bg_img, (WIDTH, HEIGHT))

piano_img = pygame.image.load('Assets/piano.png')
piano_img = pygame.transform.scale(piano_img, (318, 318))

title_img = pygame.image.load('Assets/title.png')
title_img = pygame.transform.scale(title_img, (300, 75))

start_img = pygame.image.load('Assets/start.png')
start_img = pygame.transform.scale(start_img, (180, 60))
start_rect = start_img.get_rect(center=(WIDTH//2, HEIGHT-120))

overlay = pygame.image.load('Assets/red overlay.png')
overlay = pygame.transform.scale(overlay, (WIDTH, HEIGHT))

# BUTTON IMAGES ********************************************************************

close_img = pygame.image.load('Assets/closeBtn.png')
replay_img = pygame.image.load('Assets/replay.png')
sound_off_img = pygame.image.load("Assets/soundOffBtn.png")
sound_on_img = pygame.image.load("Assets/soundOnBtn.png")

# 게임 창이 중앙에 오도록 계산
game_x = (width - WIDTH) // 2
game_y = (height - HEIGHT) // 2

# BUTTONS ********************************************************************

close_btn = Button(close_img, (36, 36), game_x + WIDTH // 4 - 27, game_y + HEIGHT//2 + 180)
replay_btn = Button(replay_img, (54,54), game_x + WIDTH // 2  - 27, game_y + HEIGHT//2 + 172)
sound_btn = Button(sound_on_img, (36, 36), game_x + WIDTH - WIDTH // 4 - 27, game_y + HEIGHT//2 + 180)

clock = pygame.time.Clock()
FPS = 30

# MUSIC **********************************************************************

buzzer_fx = pygame.mixer.Sound('Sounds/piano-buzzer.mp3')

pygame.mixer.music.load('Sounds/piano-bgmusic.mp3')
pygame.mixer.music.set_volume(0.8)
pygame.mixer.music.play(loops=-1)

# FONTS **********************************************************************

score_font = pygame.font.Font('Fonts/Futura condensed.ttf', 48)
title_font = pygame.font.Font('Fonts/Alternity-8w7J.ttf', 45)
gameover_font = pygame.font.Font('Fonts/Alternity-8w7J.ttf', 60)

title_img = title_font.render('Piano Tiles', True, WHITE)

# GROUPS & OBJECTS ***********************************************************

tile_group = pygame.sprite.Group()
square_group = pygame.sprite.Group()
text_group = pygame.sprite.Group()

time_counter = Counter(win, gameover_font, game_x, game_y)

# FUNCTIONS ******************************************************************

def get_speed(score):
	return 200 + 5 * score

def play_notes(notePath):
	pygame.mixer.Sound(notePath).play()

# NOTES **********************************************************************

with open('notes.json') as file:
	notes_dict = json.load(file)

# VARIABLES ******************************************************************

score = 0
high_score = 0
speed = 0

clicked = False
pos = None

# 롱프레스 관련 변수들
mouse_pressed = False
pressed_column = -1  # 눌리고 있는 열 번호 (-1이면 누르지 않음)
long_press_active = False
last_touched_tile_id = -1  # 마지막으로 터치된 타일의 ID
longpress_start_tile_id = -1  # 롱프레스 시작 시점의 타일 ID

home_page = True
game_page = False
game_over = False
sound_on = True

count = 0
overlay_index = 0

running = True
while running:
	pos = None

	count += 1
	if count % 100 == 0:
			square = Square(win, game_x, game_y)
			square_group.add(square)
			counter = 0

	# 전체 화면을 흰색으로 채우기
	win.fill((255, 255, 255))
	
	win.blit(bg_img, (game_x, game_y))
	square_group.update()

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE or \
				event.key == pygame.K_q:
				running = False

		if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
			# 마우스 위치를 게임 창 기준으로 변환
			pos = event.pos
			game_pos = (pos[0] - game_x, pos[1] - game_y)
			pos = game_pos
			
			# 마우스 눌림 상태 추적
			mouse_pressed = True
			if game_page and 0 <= pos[0] < WIDTH:  # 게임 창 내부인지 확인
				pressed_column = pos[0] // TILE_WIDTH  # 어느 열을 누르고 있는지 계산
				long_press_active = True
				longpress_start_tile_id = last_touched_tile_id  # 롱프레스 시작 시점 기록
		
		if event.type == pygame.MOUSEBUTTONUP:
			# 마우스 뗌 상태 처리
			mouse_pressed = False
			pressed_column = -1
			long_press_active = False

	if home_page:
		# piano.png를 게임 창 중앙에 위치시키기 (높이 조정)
		piano_rect = piano_img.get_rect()
		piano_rect.center = (game_x + WIDTH // 2, game_y + HEIGHT // 3)
		win.blit(piano_img, piano_rect)
		win.blit(start_img, (game_x + start_rect.x, game_y + start_rect.y))
		win.blit(title_img, (game_x + WIDTH // 2 - title_img.get_width() / 2 + 15, game_y + 550))

		if pos and start_rect.collidepoint(pos):
			home_page = False
			game_page = True

			# 게임 시작 시 타일 ID 카운터 초기화
			reset_tile_id_counter()
			last_touched_tile_id = -1

			x = random.randint(0, 3)
			prev_x = x
			t = Tile(x * TILE_WIDTH, -TILE_HEIGHT, win, game_x, game_y)
			tile_group.add(t)

			pos = None

			notes_list = notes_dict['2']
			note_count = 0
			pygame.mixer.set_num_channels(len(notes_list))

	if game_page:
		time_counter.update()
		if time_counter.count <= 0:
			for tile in tile_group:
				tile.update(speed)

				# 기존 클릭 처리
				if pos:
					if tile.rect.collidepoint(pos):
						if tile.alive:
							tile.alive = False
							score += 1
							if score >= high_score:
								high_score = score
							
							last_touched_tile_id = tile.tile_id  # 터치된 타일 ID 기록

							note = notes_list[note_count].strip()
							th = Thread(target=play_notes, args=(f'Sounds/{note}.ogg', ))
							th.start()
							th.join()
							note_count = (note_count + 1) % len(notes_list)

							tpos = tile.rect.centerx - 10, tile.rect.y
							text = Text('+1', score_font, tpos, win, game_x, game_y)
							text_group.add(text)

						pos = None
				
				# 롱프레스 처리: 같은 열의 연속 타일만 순서대로 자동 터치
				if long_press_active and pressed_column >= 0:
					tile_column = tile.rect.x // TILE_WIDTH  # 타일이 속한 열 계산
					if tile_column == pressed_column and tile.alive:
						# 순서 확인: 현재 터치 가능한 다음 타일인지 확인
						if tile.tile_id == last_touched_tile_id + 1:
							# 타일이 화면에 나타났고 순서가 맞으면 자동으로 터치 처리
							if tile.rect.y >= 0:  # 타일이 화면에 나타남
								tile.alive = False
								score += 1
								if score >= high_score:
									high_score = score
								
								last_touched_tile_id = tile.tile_id  # 터치된 타일 ID 기록

								note = notes_list[note_count].strip()
								th = Thread(target=play_notes, args=(f'Sounds/{note}.ogg', ))
								th.start()
								th.join()
								note_count = (note_count + 1) % len(notes_list)

								tpos = tile.rect.centerx - 10, tile.rect.y
								text = Text('+1', score_font, tpos, win, game_x, game_y)
								text_group.add(text)

				if tile.rect.bottom >= HEIGHT and tile.alive:  # 동적 게임 창 높이로 수정
					if not game_over:
						tile.color = (255, 0, 0)
						buzzer_fx.play()
						game_over = True

			if pos:
				buzzer_fx.play()
				game_over = True

			if len(tile_group) > 0:
				t = tile_group.sprites()[-1]
				if t.rect.top + speed >= 0:  # 게임 창 제일 위에서 생성
					x = random.randint(0, 3)
					if x == prev_x:
						continuous_tile = True
					else:
						continuous_tile = False
					y = -TILE_HEIGHT - (0 - t.rect.top)
					t = Tile(x * TILE_WIDTH, y, win, game_x, game_y)
					tile_group.add(t)
					prev_x = x

			text_group.update(speed)
			# Score는 왼쪽에, High 점수는 오른쪽에 배치
			img1 = score_font.render(f'Score : {score}', True, WHITE)
			win.blit(img1, (game_x + 20, game_y + 15))
			img2 = score_font.render(f'High : {high_score}', True, WHITE)
			win.blit(img2, (game_x + WIDTH - img2.get_width() - 20, game_y + 15))
			for i in range(4):
				pygame.draw.line(win, WHITE, (game_x + TILE_WIDTH * i, game_y), (game_x + TILE_WIDTH*i, game_y + HEIGHT), 1)

			speed = int(get_speed(score) * (FPS / 1000))

			if game_over:
				speed = 0

				if overlay_index > 20:
					win.blit(overlay, (game_x, game_y))

					img1 = gameover_font.render('Game over', True, WHITE)
					img2 = score_font.render(f'Score : {score}', True, WHITE)
					win.blit(img1, (game_x + WIDTH // 2 - img1.get_width() / 2, game_y + 270))
					win.blit(img2, (game_x + WIDTH // 2 - img2.get_width() / 2, game_y + 375))

					if close_btn.draw(win):
						running = False

					if replay_btn.draw(win):
						index = random.randint(1, len(notes_dict))
						notes_list = notes_dict[str(index)]
						note_count = 0
						pygame.mixer.set_num_channels(len(notes_list))

						text_group.empty()
						tile_group.empty()
						score = 0
						speed = 0
						overlay_index = 0
						game_over = False
						
						# 롱프레스 상태 초기화
						mouse_pressed = False
						pressed_column = -1
						long_press_active = False
						last_touched_tile_id = -1
						longpress_start_tile_id = -1
						
						# 타일 ID 카운터 초기화
						reset_tile_id_counter()

						time_counter = Counter(win, gameover_font, game_x, game_y)

						x = random.randint(0, 3)
						t = Tile(x * TILE_WIDTH, -TILE_HEIGHT, win, game_x, game_y)
						tile_group.add(t)

					if sound_btn.draw(win):
						sound_on = not sound_on
				
						if sound_on:
							sound_btn.update_image(sound_on_img)
							pygame.mixer.music.play(loops=-1)
						else:
							sound_btn.update_image(sound_off_img)
							pygame.mixer.music.stop()
				else:
					overlay_index += 1
					if overlay_index % 3 == 0:
						win.blit(overlay, (game_x, game_y))

	pygame.draw.rect(win, BLUE, (game_x, game_y, WIDTH, HEIGHT), 2)
	clock.tick(FPS)
	pygame.display.update()

pygame.quit()