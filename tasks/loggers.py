import logging
import sys


# create logger that writes messages to text file and stdout
log = logging.getLogger('example_log')

fmt='%(asctime)s | MY LOG MESSAGE IS: %(message)s'
format = logging.Formatter(fmt, datefmt='%m/%d/%Y %I:%M:%S %p')

handler1 = logging.FileHandler('log.txt', mode='w')
handler2 = logging.StreamHandler(sys.stderr)

handler1.setFormatter(format)
handler2.setFormatter(format)

log.addHandler(handler1)
log.addHandler(handler2)

log.setLevel(logging.INFO)

log.debug('tiny detail')
log.info('the answer is 42')
log.warning('risky stuff going on')
log.error('oops!')
