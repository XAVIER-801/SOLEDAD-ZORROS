@echo off
echo ========================================================
echo INICIANDO TUNEL DE PRUEBA PARA ACCESO EXTERNO
echo ========================================================
echo.
echo Este tunel te permitira abrir el sistema web y enviar
echo el feed de camara desde el exterior o ver la app en tu celular.
echo.
echo Se esta utilizando Pinggy.io a traves de SSH (no requiere instalacion).
echo Copia la URL publica (https://...) que aparezca abajo.
echo.
echo Presiona Ctrl+C para detener el tunel.
echo.
ssh -p 443 -R0:localhost:8000 qr@a.pinggy.io
