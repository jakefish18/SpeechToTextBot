import time
from bot.transcriber import transcribe

if __name__ == '__main__':
    start = time.time()
    text = transcribe('test_data/age_physo.mp3')

    print(text)
    end = time.time()
    print(f'Total processing time {(end - start):.3f} sec')
