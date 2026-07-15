Start-Process -NoNewWindow -FilePath "C:\Program Files\Go\bin\go.exe" -ArgumentList "run", "main.go", "-type=shard", "-port=50051", "-index-type=hnsw"
Start-Process -NoNewWindow -FilePath "C:\Program Files\Go\bin\go.exe" -ArgumentList "run", "main.go", "-type=shard", "-port=50052", "-index-type=hnsw"
Start-Process -NoNewWindow -FilePath "C:\Program Files\Go\bin\go.exe" -ArgumentList "run", "main.go", "-type=shard", "-port=50053", "-index-type=hnsw"
Start-Sleep -Seconds 2
Start-Process -NoNewWindow -FilePath "C:\Program Files\Go\bin\go.exe" -ArgumentList "run", "main.go", "-type=coordinator", "-port=50050", "-http-port=8080", "-shards=`"localhost:50051,localhost:50052,localhost:50053`""
Write-Output "Cluster started."
