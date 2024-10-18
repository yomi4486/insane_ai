@echo off
cd /d %~dp0
start uvicorn api:app --host 0.0.0.0 --port 50064 --reload