import os
import datetime
import subprocess

# Настройки
REPO_PATH = "/Users/aleksey/ProjectsVS/Tasks"
FILE_NAME = "daily_log.md"


def make_daily_commit():
    # Переходим в репозиторий
    os.chdir(REPO_PATH)

    # Получаем текущее время
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Записываем в файл
    with open(FILE_NAME, "a") as f:  # "a" - добавление в конец файла
        f.write(f"- Auto-commit: {date_str}\n")

    # Git команды
    subprocess.run(["git", "add", FILE_NAME])
    subprocess.run(["git", "commit", "-m", f"Daily commit: {date_str}"])
    subprocess.run(["git", "push", "origin", "main"])

    print(f"✅ Коммит создан: {date_str}")


if __name__ == "__main__":
    make_daily_commit()