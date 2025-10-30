import os
import datetime
import subprocess


REPO_PATH = "/Users/aleksey/ProjectsVS/Tasks"
FILE_NAME = "daily_log.md"


def make_daily_commit():
    try:
        # Переходим в репозиторий
        os.chdir(REPO_PATH)

        # сначала тянем изменения с GitHub
        print("🔄 Получаем свежие изменения с GitHub...")
        subprocess.run(["git", "pull", "origin", "main"])

        # Получаем текущее время
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # Записываем в файл
        with open(FILE_NAME, "a") as f:
            f.write(f"- Auto-commit: {date_str}\n")

        # Git команды
        subprocess.run(["git", "add", FILE_NAME])
        subprocess.run(["git", "commit", "-m", f"Daily commit: {date_str}"])

        # Пушим изменения
        print("Отправляем изменения на GitHub...")
        subprocess.run(["git", "push", "origin", "main"])

        print(f"✅ Коммит создан и отправлен: {date_str}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    make_daily_commit()