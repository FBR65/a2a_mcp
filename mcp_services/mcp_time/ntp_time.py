import ntplib
from datetime import datetime
import locale
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class NtpTime:
    """
    Fetches time from an NTP server and formats it in German locale.
    """

    def __init__(
        self, ntp_server="de.pool.ntp.org", ntp_version=3, locale_setting="de_DE.utf8"
    ):
        """
        Initializes the NtpTime class.

        Args:
            ntp_server (str): The NTP server address. Defaults to "de.pool.ntp.org".
            ntp_version (int): The NTP protocol version. Defaults to 3.
            locale_setting (str): The locale string for time formatting. Defaults to "de_DE.utf8".
        """
        self.ntp_server = ntp_server
        self.ntp_version = ntp_version
        self.locale_setting = locale_setting
        self.client = ntplib.NTPClient()
        self._set_locale()

    def _set_locale(self):
        """Sets the system locale for time formatting."""
        try:
            locale.setlocale(locale.LC_TIME, self.locale_setting)
            logging.info(f"Locale set to {self.locale_setting}")
        except locale.Error as e:
            logging.error(
                f"Could not set locale to {self.locale_setting}. "
                f"Make sure it is installed on your system. Error: {e}"
            )
            # Fallback or raise an error if locale is critical
            # For now, we'll log the error and proceed with the default locale

    def get_raw_time(self) -> datetime | None:
        """
        Fetches the current time from the NTP server.

        Returns:
            datetime | None: A datetime object representing the NTP time,
                             or None if an error occurred.
        """
        try:
            response = self.client.request(self.ntp_server, version=self.ntp_version)
            # Convert timestamp to datetime object
            dt = datetime.fromtimestamp(response.tx_time)
            return dt
        except ntplib.NTPException as e:
            logging.error(f"Error fetching time from NTP server {self.ntp_server}: {e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None

    def get_formatted_time(self, format_string="%A, %d. %B %Y, %H:%M:%S") -> str | None:
        """
        Fetches the NTP time and formats it using the configured locale.

        Args:
            format_string (str): The strftime format string.
                                 Defaults to German format "%A, %d. %B %Y, %H:%M:%S".

        Returns:
            str | None: The formatted time string, or None if time could not be fetched.
        """
        dt = self.get_raw_time()
        if dt:
            try:
                # Ensure locale is set before formatting (in case it failed initially)
                self._set_locale()
                return dt.strftime(format_string)
            except ValueError as e:
                logging.error(
                    f"Error formatting time: {e}. Check locale and format string."
                )
                # Fallback to default formatting if locale fails
                return dt.isoformat()
            except Exception as e:
                logging.error(f"An unexpected error occurred during formatting: {e}")
                return None
        else:
            return None


# --- Example Usage ---
if __name__ == "__main__":
    ntp_time_checker = NtpTime()  # Uses defaults: "de.pool.ntp.org", "de_DE.utf8"

    # Get and print the formatted German time
    german_time = ntp_time_checker.get_formatted_time()
    if german_time:
        print(f"Aktuelle NTP-Zeit (Deutsch): {german_time}")
    else:
        print("Konnte die NTP-Zeit nicht abrufen oder formatieren.")

    # Example with a different server or format
    # ntp_time_checker_us = NtpTime(ntp_server="pool.ntp.org", locale_setting="en_US.utf8")
    # us_time = ntp_time_checker_us.get_formatted_time("%A, %B %d, %Y, %I:%M:%S %p")
    # if us_time:
    #     print(f"Current NTP Time (US): {us_time}")
