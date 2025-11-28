"""
Market hours checker for Forex markets
Prevents trading when market is closed
"""

from datetime import datetime
from typing import Dict
import pytz


class MarketHoursChecker:
    """Check if market is currently open"""

    @staticmethod
    def is_forex_market_open() -> bool:
        """
        Check if Forex market is open.

        Forex market is open 24/5:
        - Opens: Sunday 22:00 GMT
        - Closes: Friday 22:00 GMT
        - Closed: Saturday 00:00 GMT to Sunday 22:00 GMT

        Returns:
            True if market is open, False if closed
        """
        utc_now = datetime.now(pytz.UTC)
        weekday = utc_now.weekday()  # 0=Monday, 6=Sunday
        hour = utc_now.hour

        # Sunday before 22:00 GMT - CLOSED
        if weekday == 6 and hour < 22:
            return False

        # Saturday - CLOSED
        if weekday == 5:
            return False

        # Friday after 22:00 GMT - CLOSED
        if weekday == 4 and hour >= 22:
            return False

        # Monday-Friday (and Sunday after 22:00) - OPEN
        return True

    @staticmethod
    def get_market_status() -> Dict:
        """
        Get detailed market status.

        Returns:
            Dict with market status information:
                - is_open: bool
                - current_time_utc: str (ISO format)
                - weekday: str
                - hour_utc: int
                - reason: str
                - next_open: str (when market opens next, if closed)
        """
        utc_now = datetime.now(pytz.UTC)
        weekday = utc_now.weekday()
        hour = utc_now.hour
        is_open = MarketHoursChecker.is_forex_market_open()

        # Calculate next open time if closed
        next_open = None
        if not is_open:
            if weekday == 6 and hour < 22:  # Sunday before 22:00
                next_open = "Sunday 22:00 GMT"
            elif weekday == 5:  # Saturday
                next_open = "Sunday 22:00 GMT"
            elif weekday == 4 and hour >= 22:  # Friday after 22:00
                next_open = "Sunday 22:00 GMT"

        # Determine reason
        if is_open:
            reason = "Market open (Forex 24/5)"
        else:
            if weekday in [5, 6]:  # Weekend
                reason = "Market closed (Weekend)"
            else:
                reason = "Market closed (Friday night to Sunday night)"

        return {
            'is_open': is_open,
            'current_time_utc': utc_now.isoformat(),
            'weekday': utc_now.strftime('%A'),
            'hour_utc': hour,
            'reason': reason,
            'next_open': next_open
        }

    @staticmethod
    def wait_time_until_open() -> int:
        """
        Calculate how long to wait until market opens (in seconds).

        Returns:
            Seconds until market opens, or 0 if already open
        """
        if MarketHoursChecker.is_forex_market_open():
            return 0

        utc_now = datetime.now(pytz.UTC)
        weekday = utc_now.weekday()
        hour = utc_now.hour

        # Calculate hours until Sunday 22:00 GMT
        if weekday == 6:  # Sunday
            hours_until_open = 22 - hour
        elif weekday == 5:  # Saturday
            hours_until_open = 24 + (22 - hour)  # Rest of Saturday + to 22:00 Sunday
        else:  # Friday evening
            days_until_sunday = (6 - weekday) % 7
            hours_until_sunday = days_until_sunday * 24
            hours_until_open = hours_until_sunday + (22 - hour)

        return max(0, hours_until_open * 3600)  # Convert to seconds


# Test function
if __name__ == "__main__":
    status = MarketHoursChecker.get_market_status()

    print("=" * 60)
    print("MARKET HOURS STATUS")
    print("=" * 60)
    print(f"Current Time (UTC): {status['current_time_utc']}")
    print(f"Weekday: {status['weekday']}")
    print(f"Hour (UTC): {status['hour_utc']}")
    print(f"\nMarket Open: {'✅ YES' if status['is_open'] else '❌ NO'}")
    print(f"Reason: {status['reason']}")

    if not status['is_open'] and status['next_open']:
        wait_seconds = MarketHoursChecker.wait_time_until_open()
        wait_hours = wait_seconds / 3600
        print(f"\nNext Open: {status['next_open']}")
        print(f"Wait Time: {wait_hours:.1f} hours ({wait_seconds} seconds)")

    print("=" * 60)
