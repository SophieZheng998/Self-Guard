import pytz
from datetime import datetime

get_time = lambda: datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific')).strftime('%y%m%d-%H%M')