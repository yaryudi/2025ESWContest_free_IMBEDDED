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

# 무지개 색상 배열 (콤보 텍스트용)
RAINBOW_COLORS = [
    (255, 0, 0),      # 빨간색 (1combo)
    (255, 165, 0),    # 주황색 (2combo)
    (255, 255, 0),    # 노란색 (3combo)
    (0, 255, 0),      # 초록색 (4combo)
    (0, 0, 255),      # 파란색 (5combo)
    (75, 0, 130),     # 남색 (6combo)
    (238, 130, 238),  # 보라색 (7combo)
]

# IMAGES *********************************************************************

bg_img = pygame.image.load('Assets/bg.png')
bg_img = pygame.transform.scale(bg_img, (WIDTH, HEIGHT))

piano_img = pygame.image.load('Assets/piano.png')
piano_img = pygame.transform.scale(piano_img, (318, 318))

title_img = pygame.image.load('Assets/title.png')
title_img = pygame.transform.scale(title_img, (300, 75))

start_img = pygame.image.load('Assets/start.png')
start_img = pygame.transform.scale(start_img, (360, 120))
start_rect = start_img.get_rect(center=(WIDTH//2, HEIGHT-120))

overlay = pygame.image.load('Assets/red overlay.png')
overlay = pygame.transform.scale(overlay, (WIDTH, HEIGHT))

# 곡 선택 화면용 이미지들
song_select_bg = pygame.image.load('Assets/bg.png')
song_select_bg = pygame.transform.scale(song_select_bg, (WIDTH, HEIGHT))

# 곡 선택 버튼들 (실제 이미지 파일들 사용)
song_buttons = []
song_images = [
    ('twinkle_twinkle', 'Assets/twinkle_twinkle.png'),
    ('HBD', 'Assets/HBD.png'),
    ('three_bears', 'Assets/three_bears.png'),
    ('airplane', 'Assets/airplane.png')
]
song_spacing = HEIGHT // 5 * 5 # 높이 5의 배수로 맞추기

for i, (song_name, image_path) in enumerate(song_images):
    # 실제 이미지 파일 로드
    button_img = pygame.image.load(image_path)
    # 이미지 크기 조정 (1.5배 크게)
    button_img = pygame.transform.scale(button_img, (375, 105))  # 250*1.5, 70*1.5
    # 위치를 중앙으로 조정 (위에서부터 시작하되 중앙에 배치)
    button_rect = button_img.get_rect(center=(WIDTH//2, song_spacing//5 * i + 175))
    song_buttons.append((button_img, button_rect, song_name))

# BUTTON IMAGES ********************************************************************

close_img = pygame.image.load('Assets/closeBtn.png')
# close 버튼 이미지를 검정색으로 변경
close_img.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)

replay_img = pygame.image.load('Assets/replay.png')
sound_off_img = pygame.image.load("Assets/soundOffBtn.png")
sound_on_img = pygame.image.load("Assets/soundOnBtn.png")
menu_img = pygame.image.load('Assets/menu.png')

# 게임 창이 중앙에 오도록 계산
game_x = (width - WIDTH) // 2
game_y = (height - HEIGHT) // 2

# BUTTONS ********************************************************************

# 버튼들을 대칭적으로 배치: replay는 중앙, close와 sound는 양쪽에 같은 거리로
button_spacing = 120  # 버튼들 사이의 거리
close_btn = Button(close_img, (36, 36), game_x + WIDTH - 60, game_y + 20)  # 오른쪽 위 모서리
replay_btn = Button(replay_img, (72, 72), game_x + WIDTH // 2 - button_spacing - 36 - 20, game_y + HEIGHT//2 + 180 + 25)
menu_btn = Button(menu_img, (120, 120), game_x + WIDTH // 2 - 60, game_y + HEIGHT//2 + 180)
sound_btn = Button(sound_on_img, (72, 72), game_x + WIDTH // 2 + button_spacing - 36 + 20, game_y + HEIGHT//2 + 180 + 25)

clock = pygame.time.Clock()
FPS = 30

# MUSIC **********************************************************************

buzzer_fx = pygame.mixer.Sound('Sounds/piano-buzzer.mp3')
buzzer_fx.set_volume(0.5)  # 부저 소리 볼륨을 30%로 설정

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
	sound = pygame.mixer.Sound(notePath)
	sound.set_volume(0.2)  # 볼륨을 0.3으로 설정 (0.0~1.0, 0.3은 30%)
	sound.play()

# NOTES **********************************************************************

with open('notes.json') as file:
	notes_dict = json.load(file)

# VARIABLES ******************************************************************

score = 0
# 각 곡별로 별도의 최고 점수 저장
high_score = {
    '1': 0,  # twinkle_twinkle
    '2': 0,  # HBD (Happy Birthday)
    '3': 0,  # three_bears
    '4': 0   # airplane
}
current_song = '1'  # 현재 선택된 곡의 키
speed = 0

clicked = False
pos = None

# 롱프레스 관련 변수들
mouse_pressed = False
pressed_column = -1  # 눌리고 있는 열 번호 (-1이면 누르지 않음)
current_column = -1  # 현재 마우스가 위치한 열 번호
long_press_active = False
drag_touch_active = False  # 드래그 터치 모드 활성화
last_touched_tile_id = -1  # 마지막으로 터치된 타일의 ID
longpress_start_tile_id = -1  # 롱프레스 시작 시점의 타일 ID
longpress_timer = 0  # 롱프레스 타이머 (일관된 속도 유지용)
longpress_delay = 10  # 롱프레스 간격 (프레임 단위, 낮을수록 빠름)
combo_count = 0  # 콤보 카운터
combo_start_tile_id = -1  # 콤보 시작 타일 ID
last_touched_column = -1  # 마지막으로 터치된 열 번호

home_page = True
song_select_page = False  # 곡 선택 화면
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
				current_column = pressed_column  # 현재 열을 눌린 열로 설정
				long_press_active = True
				longpress_start_tile_id = last_touched_tile_id  # 롱프레스 시작 시점 기록
		
		if event.type == pygame.MOUSEBUTTONUP:
			# 마우스 뗌 상태 처리
			mouse_pressed = False
			pressed_column = -1
			current_column = -1
			long_press_active = False
			drag_touch_active = False
		
		# 마우스 이동 이벤트 처리 (드래그 감지)
		if event.type == pygame.MOUSEMOTION and mouse_pressed and game_page:
			# 마우스 위치를 게임 창 기준으로 변환
			mouse_pos = event.pos
			game_mouse_pos = (mouse_pos[0] - game_x, mouse_pos[1] - game_y)
			
			# 게임 창 내부에 있고 유효한 위치인지 확인
			if 0 <= game_mouse_pos[0] < WIDTH and 0 <= game_mouse_pos[1] < HEIGHT:
				new_column = game_mouse_pos[0] // TILE_WIDTH
				
				# 열이 바뀌었으면 드래그 터치 모드 활성화
				if new_column != current_column and new_column != pressed_column:
					current_column = new_column
					drag_touch_active = True

	if home_page:
		# piano.png를 게임 창 중앙에 위치시키기 (높이 조정)
		piano_rect = piano_img.get_rect()
		piano_rect.center = (game_x + WIDTH // 2, game_y + HEIGHT // 3)
		win.blit(piano_img, piano_rect)
		win.blit(start_img, (game_x + start_rect.x, game_y + start_rect.y))
		win.blit(title_img, (game_x + WIDTH // 2 - title_img.get_width() / 2, game_y + 550))

		if pos and start_rect.collidepoint(pos):
			home_page = False
			song_select_page = True  # 곡 선택 화면으로 이동

			# 곡 선택 화면에서는 배경음악 계속 재생

	if song_select_page:
		# 곡 선택 화면 배경
		win.blit(song_select_bg, (game_x, game_y))
		
		# 곡 선택 버튼들 표시
		for button_img, button_rect, song_name in song_buttons:
			# 이미지 버튼 그리기
			win.blit(button_img, (game_x + button_rect.x, game_y + button_rect.y))
			
			# 버튼 클릭 처리
			if pos and button_rect.collidepoint(pos):
				song_select_page = False
				game_page = True
				
				# 게임 시작 시 배경음악 멈춤
				pygame.mixer.music.stop()
				
				# 선택된 곡에 따라 노래 설정
				if song_name == 'twinkle_twinkle':
					notes_list = notes_dict['1']
					current_song = '1'
				elif song_name == 'HBD':
					notes_list = notes_dict['2']
					current_song = '2'
				elif song_name == 'three_bears':
					notes_list = notes_dict['3']
					current_song = '3'
				elif song_name == 'airplane':
					notes_list = notes_dict['4']
					current_song = '4'
				
				note_count = 0
				pygame.mixer.set_num_channels(len(notes_list))
				
				# 게임 시작 시 타일 ID 카운터 초기화
				reset_tile_id_counter()
				last_touched_tile_id = -1

				x = random.randint(0, 3)
				t = Tile(x * TILE_WIDTH, -TILE_HEIGHT, win, game_x, game_y)
				tile_group.add(t)

				pos = None
				
		if close_btn.draw(win):
			running = False

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
							if score >= high_score[current_song]:
								high_score[current_song] = score
							
							# 현재 타일의 열 번호 계산
							current_tile_column = tile.rect.x // TILE_WIDTH
							
							# 콤보 처리: 열이 바뀌면 콤보 초기화
							if current_tile_column != last_touched_column:
								# 새로운 열에서 터치: 콤보 초기화
								combo_count = 0
								combo_start_tile_id = tile.tile_id
							else:
								# 같은 열에서 연속 터치: 콤보 증가
								combo_count += 1
							
							last_touched_tile_id = tile.tile_id  # 터치된 타일 ID 기록
							last_touched_column = current_tile_column  # 터치된 열 번호 기록

							note = notes_list[note_count].strip()
							th = Thread(target=play_notes, args=(f'Sounds/{note}.ogg', ))
							th.start()
							th.join()
							note_count = (note_count + 1) % len(notes_list)

							# +1 텍스트 표시
							tpos = tile.rect.centerx - 10, tile.rect.y
							text = Text('+1', score_font, tpos, win, game_x, game_y)
							text_group.add(text)
							
							# 콤보 텍스트 표시 (첫 번째 타일이 아닌 경우)
							if combo_count > 0:
								combo_text = f'{combo_count} combo'
								combo_pos = (tile.rect.centerx - 10, tile.rect.y - 30)
								# 무지개 색상 적용 (7개 색상이 반복됨)
								rainbow_color = RAINBOW_COLORS[(combo_count - 1) % len(RAINBOW_COLORS)]
								combo_text_obj = Text(combo_text, score_font, combo_pos, win, game_x, game_y, rainbow_color)
								text_group.add(combo_text_obj)

						pos = None
				
				# 롱프레스 처리: 같은 열의 연속 타일만 순서대로 자동 터치 (일관된 속도 유지)
				if long_press_active and pressed_column >= 0:
					longpress_timer += 1  # 타이머 증가
					
					# 일정 간격으로만 타일 터치 처리 (게임 속도와 무관하게 일관된 속도 유지)
					if longpress_timer >= longpress_delay:
						longpress_timer = 0  # 타이머 리셋
						
						# 현재 열에서 터치 가능한 다음 타일을 찾아서 처리
						next_tile = None
						for tile in tile_group:
							tile_column = tile.rect.x // TILE_WIDTH
							if (tile_column == pressed_column and 
								tile.alive and 
								tile.tile_id == last_touched_tile_id + 1 and
								tile.rect.y >= 0):  # 화면에 나타난 타일만
								next_tile = tile
								break
						
						# 다음 타일을 찾았으면 터치 처리
						if next_tile:
							next_tile.alive = False
							score += 1
							if score >= high_score[current_song]:
								high_score[current_song] = score
							
							# 현재 타일의 열 번호 계산
							current_tile_column = next_tile.rect.x // TILE_WIDTH
							
							# 콤보 처리: 열이 바뀌면 콤보 초기화
							if current_tile_column != last_touched_column:
								# 새로운 열에서 터치: 콤보 초기화
								combo_count = 0
								combo_start_tile_id = next_tile.tile_id
							else:
								# 같은 열에서 연속 터치: 콤보 증가
								combo_count += 1
							
							last_touched_tile_id = next_tile.tile_id  # 터치된 타일 ID 기록
							last_touched_column = current_tile_column  # 터치된 열 번호 기록

							note = notes_list[note_count].strip()
							th = Thread(target=play_notes, args=(f'Sounds/{note}.ogg', ))
							th.start()
							th.join()
							note_count = (note_count + 1) % len(notes_list)

							# +1 텍스트 표시
							tpos = next_tile.rect.centerx - 10, next_tile.rect.y
							text = Text('+1', score_font, tpos, win, game_x, game_y)
							text_group.add(text)
							
							# 콤보 텍스트 표시 (첫 번째 타일이 아닌 경우)
							if combo_count > 0:
								combo_text = f'{combo_count} combo'
								combo_pos = (next_tile.rect.centerx - 10, next_tile.rect.y - 30)
								# 무지개 색상 적용 (7개 색상이 반복됨)
								rainbow_color = RAINBOW_COLORS[(combo_count - 1) % len(RAINBOW_COLORS)]
								combo_text_obj = Text(combo_text, score_font, combo_pos, win, game_x, game_y, rainbow_color)
								text_group.add(combo_text_obj)
				
				# 드래그 터치 처리: 드래그 중 현재 열의 다음 순서 타일 자동 터치 (일관된 속도 유지)
				if drag_touch_active and current_column >= 0:
					# 현재 열에서 터치 가능한 다음 타일을 찾아서 처리
					next_tile = None
					for tile in tile_group:
						tile_column = tile.rect.x // TILE_WIDTH
						if (tile_column == current_column and 
							tile.alive and 
							tile.tile_id == last_touched_tile_id + 1 and
							tile.rect.y >= 0):  # 화면에 나타난 타일만
							next_tile = tile
							break
					
					# 다음 타일을 찾았으면 터치 처리
					if next_tile:
						next_tile.alive = False
						score += 1
						if score >= high_score[current_song]:
							high_score[current_song] = score
						
						# 현재 타일의 열 번호 계산
						current_tile_column = next_tile.rect.x // TILE_WIDTH
						
						# 콤보 처리: 열이 바뀌면 콤보 초기화
						if current_tile_column != last_touched_column:
							# 새로운 열에서 터치: 콤보 초기화
							combo_count = 0
							combo_start_tile_id = next_tile.tile_id
						else:
							# 같은 열에서 연속 터치: 콤보 증가
							combo_count += 1
						
						last_touched_tile_id = next_tile.tile_id  # 터치된 타일 ID 기록
						last_touched_column = current_tile_column  # 터치된 열 번호 기록
						drag_touch_active = False  # 이 열의 터치 완료, 다음 열 이동 대기

						note = notes_list[note_count].strip()
						th = Thread(target=play_notes, args=(f'Sounds/{note}.ogg', ))
						th.start()
						th.join()
						note_count = (note_count + 1) % len(notes_list)

						# +1 텍스트 표시
						tpos = next_tile.rect.centerx - 10, next_tile.rect.y
						text = Text('+1', score_font, tpos, win, game_x, game_y)
						text_group.add(text)
						
						# 콤보 텍스트 표시 (첫 번째 타일이 아닌 경우)
						if combo_count > 0:
							combo_text = f'{combo_count} combo'
							combo_pos = (next_tile.rect.centerx - 10, next_tile.rect.y - 30)
							# 무지개 색상 적용 (7개 색상이 반복됨)
							rainbow_color = RAINBOW_COLORS[(combo_count - 1) % len(RAINBOW_COLORS)]
							combo_text_obj = Text(combo_text, score_font, combo_pos, win, game_x, game_y, rainbow_color)
							text_group.add(combo_text_obj)

				if tile.rect.bottom >= HEIGHT and tile.alive:  # 동적 게임 창 높이로 수정
					if not game_over:
						tile.color = (255, 0, 0)
						game_over = True
						# 게임 오버 시 피아노 소리만 끊기 (부저 소리는 유지)
						pygame.mixer.set_num_channels(0)  # 모든 채널 중지
						pygame.mixer.set_num_channels(1)  # 부저 소리용 채널 1개만 유지
						# 게임 오버 시 배경음악 재생
						pygame.mixer.music.play(loops=-1)
						# 부저 소리 재생 (채널 설정 후)
						buzzer_fx.play()

			if pos:
				game_over = True
				# 게임 오버 시 피아노 소리만 끊기 (부저 소리는 유지)
				pygame.mixer.set_num_channels(0)  # 모든 채널 중지
				pygame.mixer.set_num_channels(1)  # 부저 소리용 채널 1개만 유지
				# 게임 오버 시 배경음악 재생
				pygame.mixer.music.play(loops=-1)
				# 부저 소리 재생 (채널 설정 후)
				buzzer_fx.play()

			if len(tile_group) > 0:
				t = tile_group.sprites()[-1]
				if t.rect.top + speed >= 0:  # 게임 창 제일 위에서 생성
					x = random.randint(0, 3)
					y = -TILE_HEIGHT - (0 - t.rect.top)
					t = Tile(x * TILE_WIDTH, y, win, game_x, game_y)
					tile_group.add(t)

			text_group.update(speed)
			# Score는 왼쪽에, High 점수는 오른쪽에 배치
			img1 = score_font.render(f'Score : {score}', True, WHITE)
			win.blit(img1, (game_x + 20, game_y + 15))
			img2 = score_font.render(f'High : {high_score[current_song]}', True, WHITE)
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
					img3 = score_font.render(f'High Score : {high_score[current_song]}', True, WHITE)
					win.blit(img1, (game_x + WIDTH // 2 - img1.get_width() / 2, game_y + 270))
					win.blit(img2, (game_x + WIDTH // 2 - img2.get_width() / 2, game_y + 375))
					win.blit(img3, (game_x + WIDTH // 2 - img3.get_width() / 2, game_y + 420))

					if close_btn.draw(win):
						running = False

					if replay_btn.draw(win):
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
						current_column = -1
						long_press_active = False
						drag_touch_active = False
						last_touched_tile_id = -1
						longpress_start_tile_id = -1
						longpress_timer = 0  # 롱프레스 타이머 초기화
						combo_count = 0  # 콤보 카운터 초기화
						combo_start_tile_id = -1  # 콤보 시작 타일 ID 초기화
						last_touched_column = -1  # 마지막 터치된 열 번호 초기화
						
						# 타일 ID 카운터 초기화
						reset_tile_id_counter()

						time_counter = Counter(win, gameover_font, game_x, game_y)

						x = random.randint(0, 3)
						t = Tile(x * TILE_WIDTH, -TILE_HEIGHT, win, game_x, game_y)
						tile_group.add(t)
						
						# 게임 재시작 시 배경음악 멈춤 (게임 중이므로)
						pygame.mixer.music.stop()

					if menu_btn.draw(win):
						# 모든 게임 상태 초기화
						text_group.empty()
						tile_group.empty()
						score = 0
						speed = 0
						overlay_index = 0
						game_over = False
						
						# 롱프레스 상태 초기화
						mouse_pressed = False
						pressed_column = -1
						current_column = -1
						long_press_active = False
						drag_touch_active = False
						last_touched_tile_id = -1
						longpress_start_tile_id = -1
						longpress_timer = 0  # 롱프레스 타이머 초기화
						combo_count = 0  # 콤보 카운터 초기화
						combo_start_tile_id = -1  # 콤보 시작 타일 ID 초기화
						last_touched_column = -1  # 마지막 터치된 열 번호 초기화
						
						# 타일 ID 카운터 초기화
						reset_tile_id_counter()
						
						# 게임 재시작 시 곡 선택 화면으로 이동
						game_page = False
						song_select_page = True
						pygame.mixer.music.play(loops=-1)  # 배경음악 재생

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