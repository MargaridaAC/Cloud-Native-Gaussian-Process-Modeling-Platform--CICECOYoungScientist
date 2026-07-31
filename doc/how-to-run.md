# Como Executar a Aplicação Web (GP Training App)

Para abrir a aplicação no seu computador:

1. Abra o terminal na pasta do projeto:
   ```bash
   cd c:\Users\almei\Desktop\projetoA
   ```

2. Inicie o servidor FastAPI usando o ambiente virtual (`venv`):
   ```bash
   .\venv\Scripts\python.exe -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
   ```

3. Abra o seu navegador web (Chrome, Edge, Firefox, etc.) e aceda ao endereço:
   ```
   http://localhost:8000
   ```


