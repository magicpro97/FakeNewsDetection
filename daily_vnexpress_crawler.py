"""
daily_crawl_scheduler.py

Usage:
  Linux/macOS:
    1. Ensure this script is executable:
         chmod +x daily_crawl_scheduler.py
    2. Run in background (e.g., using screen, tmux, or as a systemd service):
         ./daily_crawl_scheduler.py

  Windows:
    1. Create a Task Scheduler task:
         - Program/script: path\\to\\python.exe
         - Arguments: path\\to\\daily_crawl_scheduler.py
         - Trigger: Daily at 07:00

This script uses APScheduler to run the VNExpress crawler once per day at the specified time.
"""
from apscheduler.schedulers.blocking import BlockingScheduler
import logging
from vnexpress_crawler import main as crawl_main

# Optional: configure logging to file
logging.basicConfig(
    filename='daily_crawl.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def job():
    logging.info("Starting daily crawl…")
    try:
        result = crawl_main()
        logging.info(f"✅ Crawled {len(result['all_data'])} articles with a total of {len(result['categories'])} categories in {result['total_time']:.2f}s, saved in {result['CSV_FILE']}")
    except Exception as e:
        logging.exception("❌ Crawling failed")

if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="Asia/Ho_Chi_Minh")
    # Schedule job every day at 07:00
    scheduler.add_job(job, 'cron', hour=10, minute=6)
    print("Scheduler started, will crawl daily at 07:00")
    scheduler.start()