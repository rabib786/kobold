from playwright.sync_api import sync_playwright
import os
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        file_path = os.path.abspath("embd_res/klite.html")
        page.goto(f"file://{file_path}")

        # Handle Welcome Popup
        try:
            welcome_popup = page.locator("#welcomecontainer")
            if welcome_popup.is_visible():
                print("Welcome popup detected. Closing...")
                page.evaluate("document.getElementById('welcomecontainer').classList.add('hidden')")
                time.sleep(0.5)
        except:
            pass

        # 1. Verify Auto Summarize
        print("Checking Auto Summarize...")
        page.evaluate("display_settings()")
        page.evaluate("display_settings_tab(6)")
        if page.locator("#auto_summarize").count() > 0:
            print("SUCCESS: Auto Summarize checkbox found.")
        else:
            print("FAILURE: Auto Summarize checkbox NOT found.")

        # 2. Verify Character Creator fields
        print("Checking Character Creator fields...")
        page.evaluate("document.getElementById('settingscontainer').classList.add('hidden')")
        page.evaluate("document.getElementById('charactercreator').classList.remove('hidden')")

        fields = ["charcreator_cnotes", "charcreator_sysprompt", "charcreator_altgreetings"]
        all_found = True
        for field in fields:
            if page.locator(f"#{field}").count() > 0:
                print(f"SUCCESS: {field} found.")
            else:
                print(f"FAILURE: {field} NOT found.")
                all_found = False

        if all_found:
            print("SUCCESS: All Character Creator fields found.")

        # 3. Verify Swipe UI logic
        # Check if swipe_message function is defined
        is_defined = page.evaluate("typeof swipe_message === 'function'")
        if is_defined:
            print("SUCCESS: swipe_message function is defined.")
        else:
            print("FAILURE: swipe_message function is NOT defined.")

        # Check if gametext_alternatives is defined
        is_alts_defined = page.evaluate("typeof gametext_alternatives !== 'undefined'")
        if is_alts_defined:
            print("SUCCESS: gametext_alternatives is defined.")
        else:
            print("FAILURE: gametext_alternatives is NOT defined.")

        page.screenshot(path="verification/all_features.png")
        browser.close()

if __name__ == "__main__":
    run()
