CC = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra -pedantic
LDFLAGS = -lm
TARGET = dashboard
SRC = dashboard.c
API_TARGET = algoritmos_api
API_SRC = algoritmos_expr_api.c

all: $(TARGET) $(API_TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

$(API_TARGET): $(API_SRC)
	$(CC) $(CFLAGS) $(API_SRC) -o $(API_TARGET) $(LDFLAGS)

api: $(API_TARGET)

run: $(TARGET)
	./$(TARGET)

run-web: $(API_TARGET)
	python3 web_dashboard.py

clean:
	rm -f $(TARGET) $(API_TARGET)
