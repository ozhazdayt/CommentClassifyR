Sub Main()
    Dim ws As Worksheet
    
    For Each ws In ThisWorkbook.Sheets
        If ws.Name <> "All" Then
            ws.Rows(1).Delete
        End If
    Next ws
End Sub
