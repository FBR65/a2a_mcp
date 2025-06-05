import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException, NoSuchElementException
import trafilatura

# Configure logging for this module
logger = logging.getLogger(__name__)
# Optional: Reduce verbosity from Selenium/WebDriver Manager
logging.getLogger("selenium").setLevel(logging.WARNING)
logging.getLogger("webdriver_manager").setLevel(logging.WARNING)


class HeadlessBrowserExtractor:
    """
    Uses a headless Chrome browser (via Selenium) to load a webpage,
    allowing JavaScript rendering, and then extracts the main text content
    using Trafilatura.
    """

    def __init__(self, default_wait_time: int = 5):
        """
        Initializes the extractor.

        Args:
            default_wait_time (int): Default seconds to wait for page rendering
                                     if not specified in extract_text.
        """
        self.default_wait_time = default_wait_time
        self._chrome_options = self._configure_chrome_options()
        logger.info("HeadlessBrowserExtractor initialized.")

    def _configure_chrome_options(self) -> ChromeOptions:
        """Configures Chrome options for headless execution."""
        options = ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        # Suppress DevTools listening message
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        return options

    def extract_text(self, url: str, wait_time: int = None) -> str | None:
        """
        Loads the given URL in a headless browser, waits for rendering,
        and extracts the main text content.

        Args:
            url (str): The URL of the webpage.
            wait_time (int, optional): Seconds to wait after page load for JS rendering.
                                       Defaults to self.default_wait_time.

        Returns:
            str: The extracted main text, potentially cleaned.
                 Returns an empty string if Trafilatura finds no main content
                 but the page loaded successfully.
            None: If a significant error occurred during loading or extraction
                  (e.g., WebDriver error, page unreachable).
        """
        effective_wait_time = (
            wait_time if wait_time is not None else self.default_wait_time
        )
        logger.info(
            f"Attempting to load and extract text from: {url} (wait: {effective_wait_time}s)"
        )

        driver = None
        try:
            # Ensure WebDriver is managed for each call for isolation
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self._chrome_options)

            logger.debug(f"Navigating to {url}")
            driver.get(url)

            logger.debug(
                f"Waiting {effective_wait_time} seconds for potential JS rendering..."
            )
            time.sleep(effective_wait_time)

            page_source = driver.page_source
            if not page_source:
                logger.warning(f"Could not retrieve page source from {url}")
                return None  # Indicate a loading problem

            logger.debug(
                f"Page source retrieved ({len(page_source)} bytes). Extracting text with Trafilatura."
            )
            # Extract using Trafilatura
            extracted_text = trafilatura.extract(
                page_source,
                url=url,
                output_format="txt",
                include_comments=False,
                favor_recall=True,  # Keep favoring recall as in original code
            )

            if extracted_text:
                logger.info(
                    f"Text successfully extracted from {url} using Trafilatura."
                )
                # Clean whitespace
                return " ".join(extracted_text.split())
            else:
                logger.warning(
                    f"Trafilatura found no main text on {url} after rendering. Trying fallback."
                )
                # Fallback: Try body text (can be noisy)
                try:
                    body_text = driver.find_element("tag name", "body").text
                    if body_text:
                        logger.info("Using fallback: Text from body element.")
                        return " ".join(body_text.split())
                    else:
                        logger.info(f"Fallback failed: Body text is empty for {url}.")
                        return ""  # Page loaded, but no text found even in body
                except NoSuchElementException:
                    logger.warning(
                        f"Fallback failed: Could not find body element on {url}."
                    )
                    return ""  # Page loaded, but structure is unusual
                except Exception as body_e:
                    logger.error(
                        f"Error during fallback body text retrieval from {url}: {body_e}"
                    )
                    return ""  # Return empty string as Trafilatura didn't fail, just found nothing

        except WebDriverException as e:
            logger.error(f"Selenium WebDriver error for {url}: {e}")
            return None  # Indicate a browser/driver level error
        except Exception as e:
            logger.error(
                f"Unexpected error during extraction from {url} ({type(e).__name__}): {e}"
            )
            return None  # Indicate an unexpected error
        finally:
            if driver:
                logger.debug(f"Closing browser for {url}")
                driver.quit()


# --- Example Usage (Optional - for testing the class directly) ---
if __name__ == "__main__":
    # Example URL that might rely more on JS
    target_url = "https://quotes.toscrape.com/js/"

    print(f"\n--- Testing HeadlessBrowserExtractor with: {target_url} ---")
    extractor = HeadlessBrowserExtractor(default_wait_time=5)
    main_text = extractor.extract_text(target_url)

    if main_text is not None:
        print("\n--- Extracted Text: ---")
        # Print preview
        preview = main_text[:1000] + "..." if len(main_text) > 1000 else main_text
        print(preview)
        print(f"\n(Total length: {len(main_text)} characters)")
        print("-" * 40)
    else:
        print("\nExtraction failed or returned None (check logs for details).")
        print("-" * 40)
