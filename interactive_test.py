import os
import subprocess
import sys
import joblib

def check_model_status():
    """Checks if the backend pipeline weights are compiled and ready."""
    if os.path.exists("sentiment_model.pkl") and os.path.exists("vectorizer.pkl"):
        return "🟢 READY (Weights compiled)"
    return "🔴 MISSING (Run training pipeline first)"

def main_menu():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear') # Clean terminal screen
        print("=" * 55)
        print(" 🚀 CENTRAL DATA SCIENCE & NLP CONTROL ENGINE")
        print("=" * 55)
        print(f" Current Pipeline Status: {check_model_status()}")
        print("-" * 55)
        print(" [1] Run Training Pipeline (Process Dataset & Save Weights)")
        print(" [2] Run Terminal-Based Sandbox Tester (Fast Syntax Check)")
        print(" [3] Launch Visual Dashboard Web UI (Gradio Analytics Server)")
        print(" [4] Exit System")
        print("=" * 55)
        
        choice = input("Select an engineering task (1-4): ").strip()
        
        if choice == "1":
            print("\n⏳ Initializing train_model.py subprocess...")
            # Spawns train_model.py inside your active environment path
            subprocess.run([sys.executable, "train_model.py"])
            input("\nPress Enter to return to the main menu...")
            
        elif choice == "2":
            run_terminal_sandbox()
            
        elif choice == "3":
            if "MISSING" in check_model_status():
                print("\n⚠️ Cannot launch app.py! You must train the model first to generate weights.")
                input("\nPress Enter to return...")
                continue
            print("\n🌐 Launching app.py Gradio server interface...")
            print("Keep this terminal open. Press Ctrl+C inside this window to terminate the web server.")
            try:
                subprocess.run([sys.executable, "app.py"])
            except KeyboardInterrupt:
                print("\n🛑 Web server shut down safely.")
                input("\nPress Enter to return to main menu...")
                
        elif choice == "4":
            print("\nExiting Control Engine. Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Please pick a number from 1 to 4.")
            time.sleep(1.5)

def run_terminal_sandbox():
    """The original interactive testing playground logic isolated here."""
    print("\nLoading model for local terminal test...")
    try:
        model = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
    except FileNotFoundError:
        print("❌ Error: Missing model files. Run Option [1] first.")
        input("\nPress Enter to return...")
        return

    print("\n💬 Terminal Sandbox Active. Type 'exit' to quit back to menu.")
    print("-" * 55)
    while True:
        user_phrase = input("\nEnter phrase to test: ").strip()
        if user_phrase.lower() == 'exit':
            break
        if not user_phrase:
            continue
            
        # Transform and predict
        vec = vectorizer.transform([user_phrase.lower()])
        pred = model.predict(vec)[0]
        print(f"» Predicted Polarity Category: {pred.upper()}")

if __name__ == "__main__":
    main_menu()
