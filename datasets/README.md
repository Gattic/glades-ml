Definition of the CSV Format

While there are various specifications and implementations for the
CSV format (for ex. [4], [5], [6] and [7]), there is no formal
specification in existence, which allows for a wide variety of
interpretations of CSV files. 

For our datasets, however, we require a header line appearing as 
the first line of the file with the same format as normal record 
lines.  This header will contain names corresponding to the fields 
in the file and should contain the same number of fields as the 
records in the rest of the file.  For example:

```
field_name,field_name,field_name CRLF
aaa,bbb,ccc CRLF
zzz,yyy,xxx CRLF
```
