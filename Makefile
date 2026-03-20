CC = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra -pedantic
LDFLAGS = -lm
C_DIR = c
PY_DIR = python
BIN_DIR = $(C_DIR)/bin
TARGET = $(BIN_DIR)/dashboard
SRC = $(C_DIR)/dashboard.c
API_TARGET = $(BIN_DIR)/algoritmos_api
API_SRC = $(C_DIR)/algoritmos_expr_api.c

all: $(TARGET) $(API_TARGET)

$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

$(API_TARGET): $(API_SRC)
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(API_SRC) -o $(API_TARGET) $(LDFLAGS)

api: $(API_TARGET)

run: $(TARGET)
	./$(TARGET)

run-web: $(API_TARGET)
	python3 $(C_DIR)/web_dashboard.py

clean:
	rm -f $(TARGET) $(API_TARGET) $(BIN_DIR)/biseccion
