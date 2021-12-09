# import cv2
#
# cap = cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#
#
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     cv2.imshow('my frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# import pygame
# import sys
#
# pygame.init()
# pygame.mixer.init(44100, -16,2,2048)
#
# pygame.mixer.music.load('output.mp3')
# pygame.mixer.music.play()
#
#
# SONG_END = pygame.USEREVENT + 1
# pygame.mixer.music.set_endevent(SONG_END)
#
# # while True:
# #     pygame.mixer.music.load('output08122021143805.wav')
# #     pygame.mixer.music.play()
# #
# #     pygame.quit()
# #     sys.exit()
#
# while True:
#     for event in pygame.event.get():
#         if event.type == SONG_END:
#             print("the song ended!")
#             pygame.quit()
#             sys.exit()
# # sound_m = pygame.mixer.Sound('output.ogg')
# pygame.mixer.Sound.play(sound_m)

# mixer.music.load('output.mp3')
# mixer.music.play(1)

# import datetime
#
# x = datetime.datetime.now()
#
# print(x)
#

#
# from playsound import playsound
#
# playsound("C:/Users/dhh3hb/Documents/GitHub/BDAFA21_SSL/output.mp3")
#
# from gtts import gTTS
# import os
#
# myText = "Text To Speech Conversion Using Python"
#
# language = 'en'
#
# output = gTTS(text=myText, lang=language, slow=False)
#
# output.save("output.mp3")
#
# os.system("start output.mp3")
